import streamlit
import streamlit as st
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

def display_macro_data():
    st.subheader('Macroeconomic Data')
    # code to display macroeconomic data goes here

def display_news():
    st.subheader('News')
    # code to display news goes here

def display_fundamental_analysis():
    st.subheader('Fundamental Analysis')
    # code to display fundamental analysis goes here

def get_price_data(ticker):
    try:
        data = yf.download(ticker, period='1y', interval='1d')
    except Exception as e:
        st.error(f"Error retrieving data for {ticker}: {e}")
        data = pd.DataFrame()
    return data
    # code to get prices from ticker between an interval of time

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
    return round(current_price, 2), round(day_change, 2), round(week_change, 2), round(month_change, 2), round(year_change, 2)

def main():
    st.set_page_config(page_title='Ticker Reports', page_icon=':chart_with_upwards_trend:')
    st.title('ZIGURAT CAPITAL - FINANCIAL TOOL')

    # Add sidebar with options
    option = st.sidebar.selectbox("Select an option", ["Macro", "News", "Fundamental Analysis"])

    # If Macro is selected, display data
    if option == "Macro":
        st.subheader("MACRO ANALYSIS")
        st.write("")
        file = st.file_uploader('Upload Excel file', type=['xlsx'])
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
                                "studies": [
                                    {
                                        "id": "Moving Average",
                                        "inputs": {
                                            "length": 50,
                                            "color": "#ff6d00"
                                        }
                                    },
                                    {
                                        "id": "Moving Average",
                                        "inputs": {
                                            "length": 200,
                                            "color": "#ffd600"
                                        }
                                    }
                                ]
                            }
                        );
                    </script>
                </div>

                """ % graph

                components.html(html_code, width= 704, height=480)

    # If News or Fundamental Analysis is selected, display message
    else:
        st.write(f"You selected {option}. This feature is not available yet.")

if __name__ == '__main__':
    main()
