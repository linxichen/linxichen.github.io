---
title: "Note to Self: How to work with this site"
date: 2024-06-19T15:34:30-04:00
categories:
  - Blog
tags:
  - jekyll
  - ruby
---

## Local Develop and Preview
Update Gems
```sh
bundle update
```

After editting, run the following to see local
```sh
bundle exec jekyll serve --watch --livereload
```

Build with trace to see errors:
```sh
bundle exec jekyll build --trace
```

## Customization
#### Adding Favicon
Follow this [post](https://peateasea.de/add-favicon-to-mm-jekyll-site/)

## Publish the Page
Just git push everything.

