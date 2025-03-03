#!/bin/bash

# Check if a branch name is provided
if [ -z "$1" ]; then
	  echo "Usage: $0 <branchname>"
	    exit 1
fi

# Assign the first argument to a variable
BRANCH_NAME=$1

# Create a tag named similarly to the branch but with the 'archive/' prefix
git tag "archive/$BRANCH_NAME" "$BRANCH_NAME"

# Delete the branch locally
git branch -D "$BRANCH_NAME"

# Delete the local reference of the remote branch ("forget" the remote branch)
git branch -d -r "origin/$BRANCH_NAME"

# Push local tags to remote
git push --tags

# Delete the remote branch
git push -d origin "$BRANCH_NAME"
# or, similarly, 'git push origin :<branchname>'
