---
layout: default
---

# Git Crash Course

## Why do I need Git? And what is Git?

[Git](https://git-scm.com/) is a source code version control system.

## Clone, commit and push!

```
$ git init
```

```
$ git clone https://github.com/PnS2018/git-hello-world
```

```
$ touch hello-world.txt
$ echo "Hello World!" > hello-world.txt
$ git add hello-world.txt
$ git commit -m "first commit"
$ git push origin master
```

## Pulling, Branching, Fetching and Merging

```
$ git pull origin master
```

```
$ git fetch
```

```
$ git checkout -b new-feature
```

```
$ git checkout new-feature
```

## Respect the flow!

[GitHub Flow](https://guides.github.com/introduction/flow/) 

## Continuous Integration (Optional)

## Further Readings

+ [Official Cheat Sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)
+ [GitHub Guides](https://guides.github.com/)
+ [Online Git tutorial](https://try.github.io/)
