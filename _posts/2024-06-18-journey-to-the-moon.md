---
title: "Journey to the Moon"
date: 2024-06-18T15:34:30-04:00
categories:
  - Blog
tags:
  - neovim
  - lua
---

I am crazy about vim and especially vim binding. However, configuring VIM in the old days of vimscript is just beyond me.

First elephant in the room is the pervasive error that the global variable `vim` is undefined. Suggested [solution][vim-global-var]

```lua
lspconfig.lua_ls.setup {
  settings = {
    Lua = {
      runtime = {
        -- Tell the language server which version of Lua you're using
        -- (most likely LuaJIT in the case of Neovim)
        version = 'LuaJIT',
      },
      diagnostics = {
        -- Get the language server to recognize the `vim` global
        globals = {
          'vim',
          'require'
        },
      },
      workspace = {
        -- Make the server aware of Neovim runtime files
        library = vim.api.nvim_get_runtime_file("", true),
      },
      -- Do not send telemetry data containing a randomized but unique identifier
      telemetry = {
        enable = false,
      },
    },
  },
}
```

[vim-global-var]: https://github.com/neovim/neovim/issues/21686#issuecomment-1522446128
