// `bun scripts/check-navbar.mjs`: exit 1 when the navbar does not match the docs sections (see navbar.mjs).
import config from "../docusaurus.config.ts";
import sidebars from "../sidebars.ts";
import { problems } from "./navbar.mjs";

const errors = problems(new URL("../docs", import.meta.url).pathname, config.themeConfig, Object.keys(sidebars));
for (const error of errors) console.log(`navbar: ${error}`);
process.exit(errors.length ? 1 : 0);
