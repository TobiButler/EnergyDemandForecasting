import streamlit as st
import streamlit.web.cli as stcli
import plotly.express as px

# def main():
#     st.title("My Interactive Data Visualization App")

#     st.header("Page 1")
#     st.write("This is the first page.")
#     # Create Plotly visualizations
#     fig = px.scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13])
#     st.plotly_chart(fig)

#     st.header("Page 2")
#     st.write("This is the second page.")
#     # Create another Plotly visualization
#     fig2 = px.bar(x=['A', 'B', 'C'], y=[4, 3, 2])
#     st.plotly_chart(fig2)

#     st.header("Page 3")
#     st.write("This is the third page.")
#     # Include interactive elements
#     selected_option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
#     st.write("You selected:", selected_option)

# if __name__ == "__main__":
#     main()

def main():
    st.title("Multi-Page Streamlit App")

    menu = ["Page 1", "Page 2", "Page 3"]
    choice = st.sidebar.selectbox("Navigate", menu)

    if choice == "Page 1":
        page1()
    elif choice == "Page 2":
        page2()
    elif choice == "Page 3":
        page3()

def page1():
    st.header("Page 1 Content")
    st.write("This is the content of Page 1.")

def page2():
    st.header("Page 2 Content")
    st.write("This is the content of Page 2.")

def page3():
    st.header("Page 3 Content")
    st.write("This is the content of Page 3.")

if __name__ == "__main__":
    main()