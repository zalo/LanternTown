# Submitting the two GLTFLoader pull requests

Nothing has been pushed. Both branches are local, in `upstream/three.js`, one commit each on three.js `dev` (457581a):

| | branch | commit | PR title | PR body |
|---|---|---|---|---|
| A | `gltfloader-msft-lod` | bd87ef4 | `GLTFLoader: Add MSFT_lod support.` | `docs/gltfloader/tier-a-PR.md` |
| B | `gltfloader-msft-lod-lazy` | 078fd88 (on top of A) | `GLTFLoader: Add setLazyLOD() and setRangeRequests().` | `docs/gltfloader/tier-b-PR.md` |

The body of a PR is its `.md` file without the first line (the `# title`). Each commit ends with a
`Co-Authored-By: Claude Code` trailer and each body with a "Generated with Claude Code" line; they were there before
this pass and were left in. Remove them before step 3 if you do not want them.

Replace `YOU` with your GitHub user name. Commands are run in `upstream/three.js` unless noted.

## Before you start

Read both descriptions, in particular the first paragraph of A and "Open questions" of both. No upstream issue asks
for built-in `MSFT_lod`, a plugin for it exists, and the contributing guide suggests asking before adding a feature. If
you would rather ask first, open an issue with the first paragraph of `tier-a-PR.md` and stop here until it is answered.

## Tier A

1. Fork https://github.com/mrdoob/three.js on GitHub (web UI, or `gh repo fork mrdoob/three.js --clone=false`).
2. Add the fork as a remote:

   ```
   git remote add fork https://github.com/YOU/three.js.git
   ```

3. Check that `dev` has not moved in a way that matters, then push the branch:

   ```
   git fetch origin dev
   git log --oneline gltfloader-msft-lod..origin/dev -- examples/jsm/loaders/GLTFLoader.js test/unit/addons/loaders/GLTFLoader.tests.js
   git push fork gltfloader-msft-lod
   ```

   If the `git log` prints commits, rebase first (`git rebase origin/dev gltfloader-msft-lod`), run
   `npx eslint examples/jsm/loaders/GLTFLoader.js test/unit/addons/loaders/GLTFLoader.tests.js`, and rebase B onto the
   new A (`git rebase --onto gltfloader-msft-lod bd87ef4 gltfloader-msft-lod-lazy`).

4. Open the pull request against `mrdoob/three.js`, base branch `dev` (not `master`):

   ```
   tail -n +3 ../../docs/gltfloader/tier-a-PR.md | gh pr create --repo mrdoob/three.js --base dev --head YOU:gltfloader-msft-lod --title "GLTFLoader: Add MSFT_lod support." --body-file -
   ```

   or in the web UI: "Compare & pull request" on the fork, base `mrdoob/three.js` `dev`, the title above, the body
   pasted from `tier-a-PR.md` from the line `Related issue:` on.

5. Watch the checks of the PR. The unit tests (`npm run test-unit-addons`) and the e2e screenshots run there for the
   first time; locally only the test file under Node and `eslint` were run. No `build/` files are in the diff and none
   should be added.

## Tier B

Wait for a first reaction to A before opening B. B is three times the size, and it is pointless if A is declined.

6. In `docs/gltfloader/tier-b-PR.md` replace `#XXXX` (line 5) with the number of the Tier A pull request.
7. Push the branch:

   ```
   git push fork gltfloader-msft-lod-lazy
   ```

8. Open the pull request, again against `mrdoob/three.js` `dev`, as a draft for as long as A is open:

   ```
   tail -n +3 ../../docs/gltfloader/tier-b-PR.md | gh pr create --repo mrdoob/three.js --base dev --head YOU:gltfloader-msft-lod-lazy --draft --title "GLTFLoader: Add setLazyLOD() and setRangeRequests()." --body-file -
   ```

   Until A is merged this PR shows both commits; the description says so.

9. When A is merged, take its commit out of B and mark B as ready:

   ```
   git fetch origin dev
   git rebase --onto origin/dev gltfloader-msft-lod gltfloader-msft-lod-lazy
   git push --force-with-lease fork gltfloader-msft-lod-lazy
   ```

   If A was changed during its review, do the same rebase onto the updated A first and resolve what conflicts; B edits
   `createLOD()` and `GLTFLodExtension.loadNode()` of A.

## Before pressing the button, know what was not checked

- The throttled-network numbers in the collapsed block of `tier-b-PR.md` come from an earlier revision and an earlier
  build of the scene; they are labelled as such. Delete the block if you do not want unrepeated numbers in the PR.
- The numbers outside that block were measured on these commits, once, on this machine.
- Not run: `npm run test-unit-addons`, e2e, any browser but Chromium, `WebGPURenderer`, a phone, a cross-origin server.
