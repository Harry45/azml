import os
import streamlit as st
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
YEARS = [2024]
EX_COLUMNS = ["Category", "Item", "Cost"]
SA_COLUMNS = ["Month", "Salary"]
CAT = [
    "Bills",
    "Groceries",
    "Transportation",
    "Dining Out",
    "Entertainment",
    "Home Maintenance",
    "Healthcare",
    "Education",
    "Savings",
    "Miscellaneous",
]
# ------------------------------
st.title("SpendSmart App")


st.sidebar.header("Inputs")

form_time = st.sidebar.form("form_time")
form_time.write("Month and Year")
month = form_time.selectbox("Month", MONTHS)
year = form_time.selectbox("Year", YEARS)
submit_time = form_time.form_submit_button()

form = st.sidebar.form("form")
form.write("Do you want to record a salary?")

salary = form.number_input("Salary")
submit = form.form_submit_button()


form_ex = st.sidebar.form("form_ex")
form_ex.write("Do you want to register an expense?")
submit_no = form_ex.form_submit_button("No")
category = form_ex.selectbox("Categories", CAT)
item = form_ex.text_input("Description of expense")
if item == "":
    form_ex.error("Please enter a description of the expense.")
cost = form_ex.number_input("Amount")
submit_ex = form_ex.form_submit_button()


form_delete = st.sidebar.form("form_delete")
form_delete.write("Do you want to delete an expense?")
idx = form_delete.number_input("Row Number", step=1, format="%i")
submit_delete = form_delete.form_submit_button()


def create_csv_expenses(folder: str, month: str, year: int):
    fname = f"{folder}/{year}/{month}.csv"
    if not os.path.isfile(fname):
        os.makedirs(f"{folder}/{year}", exist_ok=True)
        dataframe = pd.DataFrame(columns=EX_COLUMNS)
        dataframe.to_csv(fname, index=True)


def create_csv_salary(folder: str, year: int):
    fname = f"{folder}/{year}.csv"
    if not os.path.isfile(fname):
        os.makedirs(f"{folder}", exist_ok=True)
        dataframe = pd.DataFrame(columns=SA_COLUMNS)
        dataframe.to_csv(fname, index=True)


def update_salary_csv(df: pd.DataFrame, month: str, year: int, value: float):

    if not month in df["Month"].values:
        new_salary = pd.DataFrame({"Month": month, "Salary": value}, index=[0])
        df = pd.concat([df if not df.empty else None, new_salary], ignore_index=True)
        df.reset_index(drop=True, inplace=True)

    idx = df[df["Month"].str.contains(month)].index.values
    row = df.iloc[idx]
    current_value = row["Salary"].values[0]
    if value != current_value:
        df.loc[idx, "Salary"] = value
    df.to_csv(f"salary/{year}.csv")
    return df


def update_expense_csv(df: pd.DataFrame, month: str, year: int, new_df: pd.DataFrame):

    lower_cases = [v.lower() for v in df["Item"].values]
    if new_df["Item"].values[0].lower() not in lower_cases:
        df = pd.concat([df if not df.empty else None, new_df], ignore_index=True)
        df.reset_index(drop=True, inplace=True)

    df.to_csv(f"expenses/{year}/{month}.csv")
    return df


def delete_row_expense_csv(row_index, month, year):
    df = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=[0])
    df = df.drop(row_index)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(f"expenses/{year}/{month}.csv")
    return df


def add_salary(month, year, salary):

    create_csv_salary("salary", year)

    # Salary
    st.subheader(f"Salary for Year {year}")
    df_salary = pd.read_csv(f"salary/{year}.csv", index_col=[0])
    df_salary = update_salary_csv(df_salary, month, year, salary)


def display_salary(month, year):
    st.subheader(f"Salary for Year {year}")
    df_salary = pd.read_csv(f"salary/{year}.csv", index_col=[0])
    st.write(df_salary)


def add_expenses(month, year, category, item, cost):
    create_csv_expenses("expenses", month, year)
    st.subheader(f"Expenses for {month}-{year}")
    df_expenses = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=[0])
    new_item = pd.DataFrame(
        {"Category": category, "Item": item, "Cost": cost}, index=[0]
    )
    df_expenses = update_expense_csv(df_expenses, month, year, new_item)
    st.write(df_expenses)


def display_expenses(month, year):
    st.subheader(f"Expenses for {month}-{year}")
    df_expenses = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=[0])
    st.write(df_expenses)


def analysis_pie_expenses(month, year):
    """ """
    st.subheader(f"Analysis for {month}-{year}")
    df_expenses = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=[0])
    total = df_expenses[["Category", "Cost"]].groupby("Category").sum()
    st.write(total.T)

    data = total["Cost"].values
    labels = list(total.index.values)

    df_salary = pd.read_csv(f"salary/{year}.csv", index_col=[0])
    colors = sns.color_palette("pastel")
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].pie(data, labels=labels, colors=colors, autopct="%.0f%%")
    ax[1].bar(
        df_salary["Month"].values, df_salary["Salary"], width=0.5, color="dodgerblue"
    )
    st.pyplot(fig)

    salary_value = df_salary[df_salary["Month"] == month]["Salary"].values
    total_cost = total["Cost"].sum()
    available = salary_value - total_cost
    st.write(f"Total cost is {total_cost.item():.2f}")
    st.write(f"Amount available to spend is {available.item():.2f}")


if submit:
    add_salary(month, year, salary)
    display_salary(month, year)

if submit_ex:
    display_salary(month, year)
    add_expenses(month, year, category, item, cost)
    analysis_pie_expenses(month, year)

if submit_delete:
    delete_row_expense_csv(idx, month, year)
    display_salary(month, year)
    display_expenses(month, year)
    analysis_pie_expenses(month, year)

if submit_no:
    display_salary(month, year)
    display_expenses(month, year)
    analysis_pie_expenses(month, year)
