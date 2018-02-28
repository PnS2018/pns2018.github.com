---
layout: default
---

# Git Crash Course

We strongly recommend this [online Git tutorial](https://try.github.io/).
You can learn how to use `Git` in 15 steps!

## Why do I need Git? And what is Git?

[Git](https://git-scm.com/) is a source code version control system.
The main purpose of having such system is to maintain the code changes.
With Git, you can track the history, save you from a post-catastrophic situation (e.g, the precious laptop is destroyed because you eat in front of it), integrate others work, make a copy of existing work along with all the history, and many other useful features.

## Clone, commit and push!

Suppose you have a directory that is

```
$ git init
```

```
$ git clone https://github.com/PnS2018/git-hello-world
```

```bash
$ touch hello-world.txt  # create a file named hello-world.txt
$ echo "Hello World!" > hello-world.txt  # append the string "Hello World!" to the document
$ git add hello-world.txt  # Add the file so that git can track it
$ git commit -m "first commit"  # make the commit, now this change is a part of history
$ git push origin master  # push it to master branch if there is a remote server
```

## Pulling, Branching, Fetching and Merging

Say if you made some changes at your working computer and want to continue your work on your own PC (I know), however, the version of code on your PC may not agree with the latest version. In this case, you will need to pull the changes from the remote server so that you can catch up:

```
$ git pull origin master
```

So there is another "pulling" command `fetch` in `git`. In the simplest terms, `git pull` does a `git fetch` followed by a `git merge`. However, we found that `git fetch` is particularly useful when you try to pull all the remote branches to local system:

```
$ git fetch
```

Branching is an important feature of Git. Initially, you have a default branch `master`. Say you want to add some feature / fix a bug to your project (e.g., let it say something funny, make sure it does not curse anyone). Generally, the idea is to make `master` branch as stable as possible. You want to avoid any untested changes. So the way you do it is to create a branch where this branch has all the previous history. Your work on this branch will not affect the master branch.
After you properly tested this new feature, you can then merge the changes to master branch.

To create a new branch, use:

```
$ git checkout -b new-feature
```

Switch to another branch, use:

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
+ [Learn Git with Bitbucket Cloud](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud)
