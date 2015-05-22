from flask import Flask, render_template, request, session, make_response, redirect
import fixtures
app = Flask(__name__)
app.config.update(DEBUG = True)
app.secret_key = "dknOUDNn84f4onfIN3fPN3fnnfFnFii839"

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/get_fixtures/", methods=['POST'])
def get_fixtures():
    rdata = request.form
    fixt = fixtures.gen_fixtures(int(rdata['teams']), int(rdata['groups']), int(rdata['pitches']), reverse=(rdata['reverse']=="true"))
    if rdata['optpitches'] == "true":
        fixt = fixtures.optimise_pitches(fixt, 2000)
    if rdata['optrounds'] == "true":
        fixt = fixtures.optimise_rounds(fixt, 2000)
    # session['fixt'] = fixt
    fixtures.save_fixtures('templates/fixtures.csv', fixt)
    return render_template('render_fixt.html', pitches=range(1, int(rdata['pitches']) + 1), fixtures=[[(r[0][i], r[1][i]) for i in range(len(r[0]))] for r in fixt])
    return "hello"

@app.route("/download/")
def download():
    response = make_response(render_template('fixtures.csv'))
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=fixtures.csv'
    return response

if __name__ == "__main__":
    app.debug = True
    app.run()
