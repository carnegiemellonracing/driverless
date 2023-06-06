Git and Github
=============================

Important Notice
-----------------

If you would like to access the repo from your command line, you need a personal access token according to Github's new requirements:

https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/

You can follow this guide:

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

Commands
--------
- Basic commands
```shell
#Lets you work in main branch
git checkout main
#Adds files with changes you made (prepare files to commit & push)
git add file1 file2
#Adds all files with changes in your current directory (prepare files to commit & push)
git add .
#Commits all of the changes with a descriptive message
git commit -m "your message"
#Pushes all of your changes to the remote repo
git push
#Fetch and merge all changes made by others from the remote repo to your local copy
git pull

#Start writing in your development branch
git checkout dev-branch
#OR you can combine the two
git checkout -b dev-branch
```
- When working on repo, it's helpful to create a new branch so you have a copy of working code saved (locally and remotely) and then changing the code on the new branch. (Creating a branch is like creating a copy of all the code in your main branch)
```shell
#Start in branch you want to copy from (usually main)
git checkout main
#Create a development branch if one does not already exist
git branch dev-branch
#Start writing in your development branch
git checkout dev-branch
#OR you can combine the two
git checkout -b dev-branch
```
- If you are in a development branch and want to revert a modified file that was committed, one way is by reverting it to its unmodified version from main
```shell
#start in your development branch
git checkout dev-branch
#replace a file with the unmodified version in your main branch
git restore -s main foldername/filename
```
- Once you're done working with a development branch (merged, messed up, ), you can delete it from both your local repo and remote repo from the command line. It can also be done through the browser on your account.
```shell
#Removing a local branch
git branch -d dev-branch
#If Git refuses to let you, you can force it with D
git branch -D dev-branch
#Removing a branch from your remote repo
git push -d origin dev-branch
#Or
git push origin --delete dev-branch
```
- To set an upstream branch while pushing changes to a new development branch
```shell
git push -u origin dev-branch
```

Extra Notes
------------
- IF YOU END UP WITH MERGE CONFLICTS: and you use VSCode, there is a UI that will let you choose which version to keep or to merge the two versions
- Make sure to comment on your commit so the person reviewing it has a clear idea of the purpose
- Make sure your local repo is up to date with the upstream to avoid merge conflicts. 
- The `.gitignore` file is used for untracked files that you want Git to ignore.
- Git stash is a way to stash changes made to a file that you don't want to commit
- Generally, it's not a good idea to commit directly to main. Instead, create a development branch and merge it with a pull request on github.
- If you are using the Zsh for your shell and want tab completion for git commands, add the following to your .zshrc file and restart the terminal:
```shell
autoload -Uz compinit && compinit
```
