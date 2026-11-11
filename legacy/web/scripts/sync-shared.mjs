// The canvas renderer is plain ES-module JavaScript shared by the Next.js app
// and the local Python server, so neither needs the other's build step. This
// copies the canonical file into public/ before dev and build.
//
// The copy in public/ is committed, so a deployment whose root directory is
// web/ still builds even when ../shared is not part of the checkout.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "shared", "board.mjs");
const dst = join(here, "..", "public", "board.mjs");

if (!existsSync(src)) {
  console.warn(`sync-shared: ${src} not found; using the committed public/board.mjs`);
  process.exit(0);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log("synced shared/board.mjs -> public/board.mjs");
