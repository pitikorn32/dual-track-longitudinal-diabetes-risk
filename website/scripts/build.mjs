import { cp, mkdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const source = fileURLToPath(new URL('../', import.meta.url));
const destination = path.join(source, 'dist');
await rm(destination, { recursive: true, force: true });
await mkdir(destination);

// Publish only the page and its curated public assets.
for (const entry of ['index.html', 'styles.css', 'site.js', 'assets']) {
  await cp(path.join(source, entry), path.join(destination, entry), { recursive: true });
}
console.log('Built website/dist');
