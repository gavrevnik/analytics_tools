from bottle import route, run, template, get, post, request, Bottle
import json

# request - инстанс с информацией, приходящей в запросе
app_print_query_params = Bottle()
@app_print_query_params.route(path='/hello', method='GET')
def hello():
    return f'Your param is {request.query.param}'

@app_print_query_params.post('/hello')
def hello():
    info = json.loads(request.json)
    return json.dumps(info['a'])


app_auth = Bottle()
def check_login(username, password):
    if username == 'n' and password == '1':
        return True
@app_auth.get('/login') # or @route('/login')
def login():
    print('hi')
    return """<form action="/login" method="post">
    Username: <input name="username" type="text" />
    Password: <input name="password" type="password" /> <input value="Login" type="submit" />
    </form>
    """
@app_auth.post('/login') # or @route('/login', method='POST')
def do_login():
    username = request.forms.get('username')
    password = request.forms.get('password')
    if check_login(username, password):
        return "<p>Your login information was correct.</p>"
    else:
        return "<p>Login failed.</p>"


# запускаем приложение
run(app=app_print_query_params, host='localhost', port=1234)
