import pandas as pd
import yfinance as yf
import streamlit as st
import requests
import streamlit.components.v1 as components
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import os
from bs4 import BeautifulSoup
import numpy as np
import altair as alt

with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=')
        os.environ[key] = value

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
        prompt=f"You are a Financial Analyst, this is the balance sheet statement from a company: Make a summary and finally give me two good things and two bad things about this statement:\n{balance_sheet}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text

    return analysis

def analyze_cash_flow(cash_flow):
    """
    Analyze cash flow data using OpenAI API and return financial analysis.
    """
    # Call OpenAI API to analyze the cash flow data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are a Financial Analyst, this is the cash flow statement from a company: Make a summary and finally give me two good things and two bad things about this statement:\n{cash_flow}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text

    return analysis

def analyze_income_statement(income):
    """
    Analyze income data using OpenAI API and return financial analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are a Financial Analyst, this is the income statement from a company: Make a summary and finally give me two good things and two bad things about this statement:\n{income}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text

    return analysis

def make_summary(variable,stock):
    """
    Analyze  data using OpenAI API and return summary financial analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are a Financial Analyst, after reading the next financial summary of {stock}, first make an analysis on the financials of {stock}, and finally give us a comentary about the sentiment on buying or selling the stock:\n{variable}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text

    return analysis

def analyze_news(input):
    """
    Analyze income data using OpenAI API and return news analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"This is a data base of latest news of a company, use data to make an analysis about how is the commpany doing and what is going to happen in the future, end the analysis with a commentary about the overall sentiment:\n{input}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def analyze_insider(input):
    """
    Analyze income data using OpenAI API and return news analysis.
    """
    # Call OpenAI API to analyze the income data
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are a top Financial Analyst, this is the information of insider trading from company, first: use this information to make an analysis about what could be happening in the company, and finally end with a commentary about the overall sentiment of the insiders:\n{input}\n\n",
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the financial analysis from OpenAI API response
    analysis = response.choices[0].text.strip()

    return analysis

def insider_trading_ticker(ti):
    # Define the URL of the page to scrape
    url = f'http://openinsider.com/screener?s={ti}&o=&pl=&ph=&ll=&lh=&fd=730&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1'

    # Send a GET request to the URL and store the response
    response = requests.get(url)

    # Parse the response with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the insider trading data
    table = soup.find('table', {'class': 'tinytable'})

    # Extract the data from the table
    data = []
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        data.append(cols)

    # Print the data
    df = pd.DataFrame(data)
    df.columns = ['x', 'Filing Date', 'Trade Date', 'Ticker', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty',
                  'Owned', 'ΔOwn', 'Value', '1d', '1w', '1m', '9m']
    df = df[['Trade Date', 'Ticker', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Owned', 'ΔOwn', 'Value']]
    df = df.drop(index=0)
    df['Value'] = df['Value'].str.replace('$', '')
    df['Value'] = df['Value'].str.replace(',', '')
    df['Value'] = pd.to_numeric(df['Value'])
    df['Type'] = np.where(df['Value'] > 0, 'buys', 'sales')
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    return df

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

def map_color(val):
    if val >= 0:
        return 'green'
    else:
        return 'red'


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
        st.text("Summary of Financials")
        st.write("")
        # Create instances of TimeSeries and FundamentalData
        ts = TimeSeries(key=api_key, output_format='pandas')
        fd = FundamentalData(key=api_key, output_format='pandas')
        # Create a text input for the stock ticker
        symbol = st.text_input('Enter Stock Ticker (e.g., AAPL):')
        if symbol:

            yahooticker = yf.Ticker(symbol)

            # Tabs for type of holder:
            slide1, slide3, slide2, slide4 = st.tabs(["Overview", "News", "Analysis", "Insider Trading"])

            with slide1:
                # Create a button to generate the report
                # Retrieve the stock data
                if symbol:
                    st.subheader("")
                    # Reques for overview
                    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
                    r = requests.get(url)

                    # Display Ticker Name:
                    tickername = r.json()['Name']
                    st.subheader(tickername)

                    # Description
                    description = r.json()['Description']
                    st.write(description)

                    st.write("Profile")

                    data = pd.DataFrame.from_dict(r.json(), orient='index').T
                    profile = data[['AssetType', 'Exchange', 'Sector', 'Industry', 'Currency', 'Country']]
                    st.dataframe(profile)

                    # Extract Holders
                    major_holders = pd.DataFrame.from_dict(yahooticker.major_holders)
                    int_holders = pd.DataFrame.from_dict(yahooticker.institutional_holders)
                    int_holders['Date Reported'] = int_holders['Date Reported'].dt.strftime('%d/%m/%Y')
                    mutual_funds = pd.DataFrame.from_dict(yahooticker.mutualfund_holders)
                    mutual_funds['Date Reported'] = mutual_funds['Date Reported'].dt.strftime('%d/%m/%Y')

                    st.write("Holders")

                    # Tabs for type of holder:
                    tab1, tab2, tab3 = st.tabs(["Major Holders", "Institutional Holders", "Mutualfund Holders"])

                    tab1.write(major_holders)
                    tab2.write(int_holders)
                    tab3.write(mutual_funds)

                    st.write("Chart")

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

            with slide2:
                if st.button("Analyze Financials"):
                    if symbol:
                        # Retrieve the balance sheet data
                        data, meta_data = fd.get_balance_sheet_annual(symbol)
                        st.write('### Balance Sheet')
                        data = data.set_index("fiscalDateEnding")
                        data = data.T
                        # OPEN AI summary request
                        resume_balance_sheet = analyze_balance_sheet(data)
                        # Display the balance sheet data summary
                        st.text(resume_balance_sheet)
                        data = data.applymap(lambda x: str(round(x/1000000, 2)) + 'm' if isinstance(x, (int, float)) else x)
                        with st.expander("Balance Sheet"):
                            st.write(data)

                        # Retrieve the cash flow data
                        data, meta_data = fd.get_cash_flow_annual(symbol)
                        st.write('### Cash Flow')
                        data = data.set_index("fiscalDateEnding")
                        data = data.T
                        # OPEN AI summary request
                        resume_cash_flow = analyze_cash_flow(data)
                        # Display the balance sheet data summary
                        st.text(resume_cash_flow)
                        numeric_cols = data.select_dtypes(include=['number']).columns
                        data = data.applymap(lambda x: str(round(x/1000000, 2)) + 'm' if isinstance(x, (int, float)) else x)
                        with st.expander("Cash Flow"):
                            st.write(data)

                        # Retrieve the income statement data
                        data, meta_data = fd.get_income_statement_annual(symbol)
                        st.write('### Income Statement')
                        data = data.set_index("fiscalDateEnding")
                        data = data.T
                        # OPEN AI summary request
                        resume_income = analyze_income_statement(data)
                        # Display the balance sheet data summary
                        st.text(resume_income)
                        numeric_cols = data.select_dtypes(include=['number']).columns
                        data = data.applymap(lambda x: str(round(x/1000000, 2)) + 'm' if isinstance(x, (int, float)) else x)
                        with st.expander("Income"):
                            st.write(data)

                        all_financials = "Balance Sheet: " + resume_balance_sheet + "Cash Flow: " + resume_cash_flow + "Income Statement: " + resume_income
                        final_summary = make_summary(all_financials, tickername)

                        st.write("Summary")
                        st.write("")
                        st.write(final_summary)
                        st.write("This is not Financial Advice, but you can contact Zigurat Capital for Extreme profits.")
                        st.write("Cheers, Financial AI.")

            with slide3:
                # Extract yfinance
                yahoonews = pd.DataFrame.from_dict(yahooticker.news)
                yahoonews = yahoonews[['title', 'relatedTickers', 'link']]
                yahoonews['link'] = yahoonews['link'].apply(
                    lambda x: f'<a href="{x}" target="_blank">🖱️</a>' if x.startswith("http") else x)

                # Reques for overview
                newsurl = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&sort=LATEST&apikey={api_key}'
                news = requests.get(newsurl)

                # Display Ticker Name:
                tickername = r.json()['Name']
                st.subheader(tickername)

                newslist = news.json()['feed']

                newsdf = pd.DataFrame.from_dict(newslist)
                newsdf = newsdf[['title', "summary", "overall_sentiment_label", "authors"]]

                news_summary = analyze_news(newsdf)
                st.write(news_summary)

                with st.expander("News Data Base"):
                    st.dataframe(newsdf)

                with st.expander("Latests News"):
                    st.markdown(yahoonews.to_html(render_links=True, escape=False), unsafe_allow_html=True, )

            with slide4:
                #Get Insider Data
                insider_data = insider_trading_ticker(symbol)
                # Display Ticker Name:
                tickername = r.json()['Name']
                st.subheader(tickername)

                insider_analysis = analyze_insider(insider_data)
                st.write(insider_analysis)

                #Make a chart
                c = alt.Chart(insider_data).mark_bar().encode(
                    x="Trade Date",
                    y="Value",
                    color=alt.condition(
                        alt.datum.Value > 0,
                        alt.value("green"),  # The positive color
                        alt.value("red")  # The negative color
                    )
                ).properties(width=600)

                st.altair_chart(c, use_container_width=True)

                with st.expander("Insider Trading Data"):
                    st.table(insider_data)

    # If News or Fundamental Analysis is selected, display message
    else:
        st.write(f"You selected {option}. This feature is not available yet.")

if __name__ == '__main__':
    main()
