# Read DB Samples
import sqlite3
import os

print('Read DB Samples')
conn = sqlite3.connect('review.sqlite')
c = conn.cursor()

SQL = "SELECT * FROM review_db"
c.execute(SQL)
results = c.fetchall()
conn.close()

print('SQL [{0}]'.format(SQL))
print('result is [{0}]'.format(results) )