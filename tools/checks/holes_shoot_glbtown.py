import json, math, base64, time, os
R = os.environ.get( 'TSQ_ROOT' ) or os.getcwd()   # (run from the project root: there is no __file__ on stdin)
T = json.load( open( R + '/out/town/town.json' ) )
poses = []
for s in T[ 'shots' ]:
    p, t = s[ 'pos' ], s[ 'target' ]
    dx, dy, dz = t[ 0 ] - p[ 0 ], t[ 1 ] - p[ 1 ], t[ 2 ] - p[ 2 ]
    poses.append( ( s[ 'name' ], p[ 0 ], p[ 1 ], math.atan2( - dx, dy ), math.atan2( dz, math.hypot( dx, dy ) ) ) )
print( [ q[ 0 ] for q in poses ] )
first = True
for f in ( 'town_lo', 'town_lo_r2', 'town_mid', 'town_mid_r2', 'town_near', 'town_near_r2' ):
    url = 'http://localhost:8765/glbtown/?pr=1&tone=none&sky=none&bg=ff00ff&file=' + f + '.glb'
    goto_url( url )
    wait_for_load()
    for i in range( 120 ):
        st = js( "JSON.stringify( window.__stats ? { d: __stats.done, e: __stats.error } : null )" )
        if st and json.loads( st ) and ( json.loads( st )[ 'd' ] or json.loads( st )[ 'e' ] ): break
        time.sleep( 1 )
    print( f, st )
    if not st or not json.loads( st ) or json.loads( st )[ 'e' ] or not json.loads( st )[ 'd' ]: continue
    for name, x, y, yaw, pitch in poses:
        js( f"( () => {{ const P = __view.P; P.x = {x}; P.y = {y}; P.yaw = {yaw}; P.pitch = {pitch}; }} )()" )
        time.sleep( 0.6 )
        d = js( "__view.shot()" )
        open( f'{R}/.scratch/main/holes/shots/{f}__{name}.jpg', 'wb' ).write( base64.b64decode( d.split( ',' )[ 1 ] ) )
goto_url( 'about:blank' )
print( 'done' )
