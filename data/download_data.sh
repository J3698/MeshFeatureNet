#!/usr/bin/env bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JRE3fzIS9q2Jvou-DZuogY9VHiTftdb_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JRE3fzIS9q2Jvou-DZuogY9VHiTftdb_" -O data/MN40Objs.zip && rm -rf /tmp/cookies.txt

echo "Silently exctracting zip..."
unzip data/MN40Objs > /dev/null
mv MN40Objs data/MN40Objs
echo "Done downloading and extracting"
