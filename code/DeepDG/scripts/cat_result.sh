#!/bin/bash
find $@ -type f -name "done.txt" | xargs -I {} sh -c 'echo "Directory: $(dirname {})" && cat {}'
