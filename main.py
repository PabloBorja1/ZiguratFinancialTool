import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from dotenv import load_dotenv
load_dotenv()
import os

# Set up OpenAI API authentication
import openai
openai.api_key = os.environ.get('closedia')

# API key for Alpha Vantage
api_key = os.environ.get('betadisvantage')

#Start definitions:

def display_macro_data():
    st.subheader('Macroeconomic Data')
    # code to display macroeconomic data goes here

def display_news():
    st.subheader('News')
    # code to display news goes here

def display_financials():
    st.subheader('Summary of Financials')
    # code to display Summary of Financials goes here

def get_news(ticker):
    sn = StockNews({ticker}, wt_key='MY_WORLD_TRADING_DATA_KEY')
    news = sn.read_rss()
    dfnews = pd.DataFrame(news, columns=["stock","title", "summary", "date","sentiment_summary"])
    return dfnews

def get_price_data(ticker):
    try:
        data = yf.download(ticker, period='1y', interval='1d')
    except Exception as e:
        st.error(f"Error retrieving data for {ticker}: {e}")
        data = pd.DataFrame()
    return data
    # code to get prices from ticker between an interval of time

def analyze_balance_sheet(balance_sheet):
    """
    Analyze balance sheet data using OpenAI API and return financial analysis.
    """
    # Call OpenAI API to analyze the balance sheet data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"This is the balance sheet data from a company, use this balance sheet to make a financial analysis and give us two good things and two bad things about this company based in the statement:\n{balance_sheet}\n\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def analyze_cash_flow(cash_flow):
    """
    Analyze cash flow data using OpenAI API and return financial analysis.
    """
    # Call OpenAI API to analyze the cash flow data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"This is the cash flow statement data from a company, use this cash flow to make a financial analysis and give us two good things and two bad things about this company based in the statement:\n{cash_flow}\n\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def analyze_income_statement(income):
    """
    Analyze income data using OpenAI API and return financial analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"This is the income statement data from a company, use this income statement to make a financial analysis and give us two good things and two bad things about this company based in the statement:\n{income}\n\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def make_summary(variable,variable2):
    """
    Analyze  data using OpenAI API and return summary financial analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Please make a summary about this financial analysis text\n{variable2}\n:\n{variable}\n\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def calculate_changes(data):
    close = data['Close']
    current_price = close.iloc[-1]
    if len(close) >= 2:
        day_change = ((current_price - close.iloc[-2]) / close.iloc[-2]) * 100
    else:
        day_change = 0
    week_change = ((current_price - close.iloc[-6]) / close.iloc[-6]) * 100
    month_change = ((current_price - close.iloc[0]) / close.iloc[0]) * 100
    if len(close) >= 23:
        year_change = ((current_price - close.iloc[-22]) / close.iloc[-22]) * 100
    else:
        year_change = 0
    return round(current_price, 2), round(day_change, 2), round(week_change, 2), round(month_change, 2), round(
        year_change, 2)

def main():
    st.set_page_config(page_title='Ticker Reports', page_icon=':chart_with_upwards_trend:')
    st.title('ZIGURAT CAPITAL - FINANCIAL TOOL')

    # Add sidebar with options
    option = st.sidebar.selectbox("Select an option", ["Summary of Financials","Macro", "News"])

    # If Macro is selected, display data
    if option == "Macro":
        st.subheader("MACRO ANALYSIS")
        st.write("")
        st.write("The excel file should have the columns Ticker, Graph(Trading View Ticker), Long Name(Name), Type(Class of the ticker)")
        file = st.file_uploader('Upload Excel file,', type=['xlsx'])
        if file is not None:
            df = pd.read_excel(file)
            types = df['Type'].unique()
            st.write("")
            st.subheader("Price Changes")
            selected_type = st.selectbox('Select a Type', types)
            tickers = df[df['Type'] == selected_type]['Ticker']
            data_dict = {}
            for ticker in tickers:
                data = get_price_data(ticker)
                current_price, day_change, week_change, month_change, year_change = calculate_changes(data)
                color = 'green' if month_change > 1 else 'red' if month_change < -1 else 'black'
                long_name = df[df['Ticker'] == ticker]['Long Name'].iloc[0]  # get long name from the DataFrame
                data_dict[ticker] = {'Long Name': long_name,  # add long name to the dictionary
                                     'Current Price': current_price,
                                     '% Day': day_change,
                                     '% Week': week_change,
                                     '% Month': month_change,
                                     '% Year': year_change}
            data_df = pd.DataFrame.from_dict(data_dict, orient='index')
            data_df = data_df[
                ['Long Name', 'Current Price', '% Day', '% Week', '% Month', '% Year']]  # rearrange the columns
            styled_data_df = data_df.style.format(
                {'Current Price': '{:.2f}', '% Day': '{:.2f}', '% Week': '{:.2f}', '% Month': '{:.2f}',
                 '% Year': '{:.2f}'}
            ).applymap(lambda x: 'color: green' if x > 1 else 'color: red' if x < -1 else '',
                       subset=pd.IndexSlice[:, ['Current Price', '% Day', '% Week', '% Month', '% Year']])
            st.write(styled_data_df)
            st.write("")
            st.subheader("Charts")
            for ticker in tickers:
                long_name = df[df['Ticker'] == ticker]['Long Name'].iloc[0]  # get long name from the DataFrame
                graph = df[df['Ticker'] == ticker]['Graph'].iloc[0]  # get symbol from the DataFrame
                st.write('##', ticker, "(", long_name, ")")

                html_code = """

                <div class="tradingview-widget-container">
                    <div id="tradingview_1"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                        new TradingView.widget(
                            {
                                "width": 704,
                                "height": 480,
                                "symbol": "%s",
                                "interval": "D",
                                "timezone": "Etc/UTC",
                                "theme": "dark",
                                "style": "1",
                                "locale": "en",
                                "toolbar_bg": "#f1f3f6",
                                "enable_publishing": false,
                                "allow_symbol_change": true,
                                "container_id": "tradingview_1",
                                
                            }
                        );
                    </script>
                </div>
                """ % graph

                components.html(html_code, width=704, height=480)

    # If Summary of Financials selected, display data
    elif option == "Summary of Financials":

        # Create instances of TimeSeries and FundamentalData
        ts = TimeSeries(key=api_key, output_format='pandas')
        fd = FundamentalData(key=api_key, output_format='pandas')
        # Create a text input for the stock ticker
        symbol = st.text_input('Enter Stock Ticker (e.g., AAPL):')

        # Create a button to generate the report
        if st.button('Generate Report'):

            # Retrieve the stock data
            if symbol:

                st.write('##', symbol)
                st.write("Graph")

                html_code = """

                <div class="tradingview-widget-container">
                    <div id="tradingview_1"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                        new TradingView.widget(
                            {
                                "width": 704,
                                "height": 480,
                                "symbol": "%s",
                                "interval": "D",
                                "timezone": "Etc/UTC",
                                "theme": "dark",
                                "style": "1",
                                "locale": "en",
                                "toolbar_bg": "#f1f3f6",
                                "enable_publishing": false,
                                "allow_symbol_change": true,
                                "container_id": "tradingview_1",

                            }
                        );
                    </script>
                </div>
                """ % symbol

                components.html(html_code, width=704, height=480)

                # Retrieve the balance sheet data
                data, meta_data = fd.get_balance_sheet_annual(symbol)
                st.write('### Balance Sheet')
                #OPEN AI summary request
                resume_balance_sheet = analyze_balance_sheet(data)
                # Display the balance sheet data summary
                st.write(resume_balance_sheet)
                st.write(data)

                # Retrieve the cash flow data
                data, meta_data = fd.get_cash_flow_annual(symbol)
                st.write('### Cash Flow')
                # OPEN AI summary request
                resume_cash_flow = analyze_cash_flow(data)
                # Display the balance sheet data summary
                st.write(resume_cash_flow)
                st.write(data)

                # Retrieve the income statement data
                data, meta_data = fd.get_income_statement_annual(symbol)
                st.write('### Income Statement')
                # OPEN AI summary request
                resume_income = analyze_income_statement(data)
                # Display the balance sheet data summary
                st.write(resume_income)
                st.write(data)

                all_financials = resume_balance_sheet + resume_cash_flow + resume_income
                final_summary = make_summary(all_financials,symbol)

                st.write("Summary")
                st.write("")
                st.write(final_summary)


    # If News or Fundamental Analysis is selected, display message
    else:
        st.write(f"You selected {option}. This feature is not available yet.")

if __name__ == '__main__':
    main()