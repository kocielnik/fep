readme.md: readme.ipynb
	jupyter nbconvert --execute --to markdown $<
