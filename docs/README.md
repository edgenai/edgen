# Edgen Docs

## Getting started

To get started developing, first install the npm dependencies:

```bash
npm install
```

Next, run the development server:

```bash
npm run dev
```

Finally, open [http://localhost:3000](http://localhost:3000) in your browser to view the website.

## Global search

The docs includes a global search that's powered by the [FlexSearch](https://github.com/nextapps-de/flexsearch) library. It's available by clicking the search input or by using the `CTRL+K` shortcut.

This feature requires no configuration, and works out of the box by automatically scanning your documentation pages to build its index. The search parameters can be adjusted by editing the `src/mdx/search.mjs` file.
