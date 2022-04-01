#! /bin/bash

# pass the switchboard text file to be cleaned
swbd_text=$1

# splitting the switchboard text into IDs and Just the text without any prefix
sed 's/ /@/1' $swbd_text | awk 'BEGIN{FS="@"} {print $1}' > ids
sed 's/ /@/1' $swbd_text | awk 'BEGIN{FS="@"} {print $2}' > just_text

# cleaning the text: removes special characters, punctuations etc. But, numbers are not transliterated
sed 's/\b-\b/ /g' just_text | sed 's/-//g' | sed 's/&/ and /g' | sed 's/\._/ /g' | sed 's/\b\.\b/ /g' | sed 's/\.//g' | sed 's/_//g' | sed 's/20\/20/twenty twenty/g' | sed 's/\[[^][]*\]//g' | tr -s ' ' | sed "s/^[ \t]*//" | awk '{$1=$1};1' > processed_just_text


# If you want to have numbers written in english(transliterated), please do that manually before proceeding with the next steps


sed 's/\(.\)/\1\n/g' processed_just_text | sort | uniq -c > counts
awk '{print $2 " " $1}' counts | sort > dict.ltr.txt

# adds prefix to the cleaned text
paste -d ' ' ids processed_just_text > lc_clean_text_swbd

# Convert from lower case to upper case
tr '[:lower:]' '[:upper:]' < lc_clean_text_swbd > clean_text_swbd
rm ids just_text processed_just_text counts lc_clean_text_swbd