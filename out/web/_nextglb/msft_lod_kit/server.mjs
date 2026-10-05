#!/usr/bin/env node
// server.mjs - a static file server for this kit that answers HTTP range requests. Node, no dependencies.
//
//   node server.mjs [--port 8791] [--root .] [--mount /three=/path/to/three.js] [--gzip] [--log]
//
//   --root DIR        what is served at / (default: the folder of this file)
//   --mount /P=DIR    serves DIR at /P/ (repeatable), e.g. a three.js checkout for check.html's ?three= and ?addons=
//   --gzip            gzip answers to requests WITHOUT a Range header when the client accepts it (as a real host
//                     would); an answer to a Range request is never encoded: ranges of a gzip stream are no use
//   --log             one line per request: status, bytes sent, Range and Accept-Encoding of the request
//
// Range: one range (bytes=a-b, bytes=a-, bytes=-n) -> 206 with Content-Range; anything else -> the whole file.
// A path that starts with /norange/ is the same file from a server that ignores Range (200, whole body).
// Every answer carries Cache-Control: no-store and permissive CORS headers. Listens on 127.0.0.1 only.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice( 2 );
const opt = { port: 8791, root: path.dirname( fileURLToPath( import.meta.url ) ), mounts: [], gzip: false, log: false };
for ( let i = 0; i < args.length; i ++ ) {

	if ( args[ i ] === '--port' ) opt.port = + args[ ++ i ];
	else if ( args[ i ] === '--root' ) opt.root = path.resolve( args[ ++ i ] );
	else if ( args[ i ] === '--mount' ) { const [ p, d ] = args[ ++ i ].split( '=' ); opt.mounts.push( [ p.replace( /\/$/, '' ) + '/', path.resolve( d ) ] ); }
	else if ( args[ i ] === '--gzip' ) opt.gzip = true;
	else if ( args[ i ] === '--log' ) opt.log = true;
	else { console.error( 'unknown option ' + args[ i ] ); process.exit( 1 ); }

}

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json', '.gltf': 'model/gltf+json', '.glb': 'model/gltf-binary', '.bin': 'application/octet-stream', '.wasm': 'application/wasm', '.md': 'text/plain; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.ktx2': 'image/ktx2', '.css': 'text/css' };
const zipped = new Map(); // file -> { mtime, body }

function resolve( urlPath ) {

	let p = decodeURIComponent( urlPath );
	let norange = false;
	if ( p.startsWith( '/norange/' ) ) { norange = true; p = p.slice( 8 ); }
	let root = opt.root;
	for ( const [ prefix, dir ] of opt.mounts ) if ( p.startsWith( prefix ) ) { root = dir; p = p.slice( prefix.length - 1 ); break; }
	const file = path.join( root, path.normalize( p ) );
	if ( file !== root && ! file.startsWith( root + path.sep ) ) return null;
	return { file, norange };

}

http.createServer( ( req, res ) => {

	const url = new URL( req.url, 'http://x' );
	const head = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Range', 'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges, Content-Encoding', 'Cross-Origin-Resource-Policy': 'cross-origin', 'Timing-Allow-Origin': '*', 'Cache-Control': 'no-store' };
	const done = ( status, sent ) => { if ( opt.log ) console.log( `${status} ${String( sent ).padStart( 9 )}  ${req.method} ${req.url}  range=${req.headers.range || '-'}  accept-encoding=${req.headers[ 'accept-encoding' ] || '-'}` ); };
	if ( req.method === 'OPTIONS' ) { res.writeHead( 204, head ); res.end(); return; }
	const at = resolve( url.pathname );
	let file = at && at.file;
	if ( file && fs.existsSync( file ) && fs.statSync( file ).isDirectory() ) file = path.join( file, 'index.html' );
	if ( ! file || ! fs.existsSync( file ) || ( req.method !== 'GET' && req.method !== 'HEAD' ) ) { res.writeHead( 404, head ); res.end( 'not found' ); done( 404, 0 ); return; }
	const stat = fs.statSync( file ), size = stat.size;
	head[ 'Content-Type' ] = TYPES[ path.extname( file ).toLowerCase() ] || 'application/octet-stream';
	if ( ! at.norange ) head[ 'Accept-Ranges' ] = 'bytes';
	const m = ! at.norange && /^bytes=(\d*)-(\d*)$/.exec( req.headers.range || '' );
	if ( m && ( m[ 1 ] !== '' || m[ 2 ] !== '' ) ) {

		let start = m[ 1 ] === '' ? Math.max( 0, size - ( + m[ 2 ] ) ) : + m[ 1 ];
		let end = m[ 1 ] === '' || m[ 2 ] === '' ? size - 1 : Math.min( + m[ 2 ], size - 1 );
		if ( start >= size || start > end ) { res.writeHead( 416, { ...head, 'Content-Range': `bytes */${size}` } ); res.end(); done( 416, 0 ); return; }
		res.writeHead( 206, { ...head, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': end - start + 1 } );
		if ( req.method === 'HEAD' ) { res.end(); return; }
		fs.createReadStream( file, { start, end } ).pipe( res );
		done( 206, end - start + 1 );
		return;

	}

	if ( opt.gzip && ! req.headers.range && /\bgzip\b/.test( req.headers[ 'accept-encoding' ] || '' ) && ! /\.(png|jpg|webp|ktx2)$/i.test( file ) ) {

		let z = zipped.get( file );
		if ( ! z || z.mtime !== stat.mtimeMs ) { z = { mtime: stat.mtimeMs, body: zlib.gzipSync( fs.readFileSync( file ), { level: 6 } ) }; zipped.set( file, z ); }
		res.writeHead( 200, { ...head, 'Content-Encoding': 'gzip', 'Vary': 'Accept-Encoding', 'Content-Length': z.body.length } );
		res.end( req.method === 'HEAD' ? undefined : z.body );
		done( 200, z.body.length );
		return;

	}

	res.writeHead( 200, { ...head, 'Content-Length': size } );
	if ( req.method === 'HEAD' ) { res.end(); return; }
	fs.createReadStream( file ).pipe( res );
	done( 200, size );

} ).listen( opt.port, '127.0.0.1', () => console.log( `msft_lod_kit: http://127.0.0.1:${opt.port}/check.html  (root ${opt.root}${opt.mounts.map( ( [ p, d ] ) => `, ${p} = ${d}` ).join( '' )})` ) );
