# Implement Web Application 2 using Flask And WTForms Library
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

class HelloForm(Form):
	# Derived from WTForm Object, only define TextAreaField here
	sayHello = TextAreaField('', [validators.DataRequired()])

@app.route('/')
def index():
	form = HelloForm(request.form)
	return render_template('first_app.html', form = form)

@app.route('/hello', methods=['POST'])
def hello():
	form = HelloForm(request.form)
	if request.method == 'POST' and form.validate():
		name = request.form['sayHello']
		return render_template('hello.html', name = name)
	else:
		return render_template('first_app.html', form = form)

if __name__ == '__main__':
	app.run(debug = True)