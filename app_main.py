from flask import Flask, render_template, request
from datetime import datetime, time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            principal = float(request.form.get('principal'))
            rate = float(request.form.get('rate'))  # This is a monthly rate
            date = datetime.strptime(request.form.get('date'), '%Y-%m-%d')
            end_date = datetime.now()  # get current date and time
            total_days = (end_date - date).days

            years = total_days // 365
            remaining_days = total_days % 365
            months = remaining_days // 30
            days = remaining_days % 30

            amount = principal
            for i in range(years):
                yearly_interest = 0
                for j in range(12):  # calculate interest for each month in a year
                    yearly_interest += amount * rate / 100
                amount += yearly_interest  # add the yearly interest to the principal

            if months > 0:
                monthly_interest = amount * rate / 100 * months  # calculate interest for remaining months
                amount += monthly_interest

            if days > 0:
                daily_interest = amount * rate / 100 / 30 * days  # calculate interest for remaining days
                amount += daily_interest

            result = f"The amount after {years} years, {months} months and {days} days is {amount}"
            return render_template('index.html', result=result)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)