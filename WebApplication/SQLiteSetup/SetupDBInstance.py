# Setup a Sample SQLite DBInstance
import sqlite3
import os

print('Setup a Sample SQLite DBInstance')
conn = sqlite3.connect('review.sqlite')
c = conn.cursor()

createTableSQL = 'CREATE TABLE review_db ' \
				 '(review TEXT, sentiment INTEGER, date TEXT)'
print('Create Table SQL: [{0}]'.format(createTableSQL))
c.execute(createTableSQL)

sample1 = 'I love this movie'
sample2 = 'I disliked this movie'
insertExampleSQL = "INSERT INTO review_db " \
					"(review, sentiment, date) VALUES" \
					"(?, ?, DATETIME('now'))"
print('Insert Exapmle 1 [{0}]; Example 2[{1}] into DB SQL\n{2}'.format(sample1, sample2, insertExampleSQL))

c.execute(insertExampleSQL, (sample1, 1))
c.execute(insertExampleSQL, (sample2, 0))
conn.commit()
conn.close()