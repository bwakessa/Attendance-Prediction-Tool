from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import secrets
from typing import TextIO
from tkinter import messagebox
import sqlite3
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statsmodels.api as sm
from sklearn import linear_model, metrics
from datetime import datetime
from math import cos, e, pi, pow

root = Tk()
root.title("Attendance Data Prediction")
root.geometry("600x250")
root.config(bg="black")

conn = sqlite3.connect("attendees.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS attendee_list ("NAME" TEXT, "EMAIL" TEXT, "PHONE_NUMBER" TEXT, "AGE" INTEGER, "GENDER" TEXT, "TOTAL_DONATIONS" INTEGER, "LARGEST_DONATION" INTEGER, "FIRST_YEAR" INTERGER, "NUM_FUNDRAISERS_ATTENDED" INTEGER)""")
c.close()
conn.commit()
conn.close()


def weight(a, g, d, f, n):
    """preconditions:
    - <a> is age: a >= 0
    - <g> is gender: 0 <= g <= 1 (0 = MALE, 1 = FEMALE)
    - <d> is total donations: d >= 0
    - <f> is year of first fundraiser ever attended: current year >= f >= current year - a
    - <n> is total number of fundraisers ever attended: a >= n >= 0"""
    if f == datetime.now().year:
        t1 = ((0.04 * a) * (1.1 * n)) / (datetime.now().year - (f + 1))
    else:
        t1 = ((0.04 * a) * (1.1 * n)) / (datetime.now().year - f)
    t2 = pow(e, ((abs(g - 0.55)) * (0.002 * d)))
    return (pi / (1 + (t1 * t2))) - (0.03 * n)


def cosmoid(x):
    z = 4 * cos(x)
    t1 = 1 + pow(e, -z)
    return 1 / t1


def load_model():
    pass
    #load data into pandas DataFrame
    conn = sqlite3.connect("attendees.db")
    c = conn.cursor()

    data = c.execute("SELECT * FROM attendee_list")

    df = pd.DataFrame(c.fetchall(), columns=['NAME', 'EMAIL', 'PHONE NUMBER', 'AGE',
                                             'GENDER', 'TOTAL DONATIONS',
                                             'LARGEST DONATION', 'FIRST YEAR',
                                             'NUM FUNDRAISERS ATTENDED'])

    df = df.astype({"GENDER": 'int64'})
    weights = pd.Series([])
    for i in range(df.shape[0]):
        w = weight(int(df.loc[i, 'AGE']),
                   int(df.loc[i, 'GENDER']),
                   int(df.loc[i, 'TOTAL DONATIONS']),
                   int(df.loc[i, 'FIRST YEAR']),
                   int(df.loc[i, 'NUM FUNDRAISERS ATTENDED']))
        weights[i] = cosmoid(w)

    #add weights to dataframe
    df.insert(df.shape[1], 'WEIGHTS', weights)

    print(df.to_string())

    #train MLR model
    model = linear_model.LinearRegression()
    model.fit(df[['AGE', 'GENDER', 'TOTAL DONATIONS', 'FIRST YEAR', 'NUM FUNDRAISERS ATTENDED']], df[['WEIGHTS']])


    n_root = Toplevel()
    n_root.resizable(False, False)
    n_root.title("Data Manager")
    n_root.geometry("1000x800")
    n_root.config(bg="gray")
    root.withdraw()

    def go_back():
        n_root.withdraw()
        root.deiconify()
    n_root.protocol("WM_DELETE_WINDOW", go_back)

    # - provide description and details of model
        # R, R^2, equation of MLR model,

    eq = 'Y = {} + {}*X_1 + {}*X_2 + {}*X_3 + {}*X_4 + {}*X_5'.format(model.intercept_,
                                                                      model.coef_[0][0],
                                                                      model.coef_[0][1],
                                                                      model.coef_[0][2],
                                                                      model.coef_[0][3],
                                                                      model.coef_[0][4])
    model_equation = Label(n_root, text=eq, bg='black', fg='white')
    model_equation.place(relx=0.06, rely=0.1)

    Label(n_root, text='Complete Model', bg='black', fg='white', font='Times 15').place(relx=0.45, rely=0.05)
    Label(n_root, text='Y = Probability Weight (Dependent)', bg='gray', font='Times 12').place(relx=0.06, rely=0.15)
    Label(n_root, text='X_1 = Age', bg='gray', font='Times 12').place(relx=0.06, rely=0.185)
    Label(n_root, text='X_2 = Gender', bg='gray', font='Times 12').place(relx=0.06, rely=0.22)
    Label(n_root, text='X_3 = Total Donations', bg='gray', font='Times 12').place(relx=0.06, rely=0.255)
    Label(n_root, text='X_4 = First Year', bg='gray', font='Times 12').place(relx=0.06, rely=0.29)
    Label(n_root, text='X_5 = Number of Fundraisers Attended', bg='gray', font='Times 12').place(relx=0.06, rely=0.325)

    #COMPUTING ADJUSTED R^2 VALUE
    m = sm.OLS(df[["WEIGHTS"]], df[['AGE', 'GENDER', 'TOTAL DONATIONS', 'FIRST YEAR', 'NUM FUNDRAISERS ATTENDED']]).fit()
    Label(n_root, text='R^2 = {}'.format(m.rsquared_adj), bg='gray', font="Times 15").place(relx=0.68, rely=0.14)

    def regression_mode(_):

        Label(n_root, text="Predictor(s)", bg='gray', fg='black', font="Times 13").place(relx=0.4, rely=0.465)
        Label(n_root, text="Dependent(s)", bg='gray', fg='black', font="Times 13").place(relx=0.53, rely=0.465)
        if s0.get() == "Simple Linear Regression":
            m1.place(relx=0.38, rely=0.5, width=122)
            m3.place(relx=0.517, rely=0.5, width=122)
            m2.place_forget()
        else:
            m1.place(relx=0.38, rely=0.5, width=122)
            m2.place(relx=0.38, rely=0.55, width=122)
            m3.place(relx=0.517, rely=0.5, width=122)

        def regressionals():
            figure = plt.figure()
            ax = figure.add_subplot(111)

            if s0.get() == "Simple Linear Regression":
                n2_root = Toplevel()
                n2_root.title("Display")
                n2_root.geometry("1000x800")
                n2_root.config(bg="white")

                sl_model = linear_model.LinearRegression()
                X, y = df[[s1.get().upper()]], df[[s3.get().upper()]]
                sl_model.fit(X, y)
                y_pred = sl_model.predict(X)

                ax.scatter(X, y, color='blue', marker='o', label='Actual')
                ax.plot(X, y_pred, color='red', marker='o', markerfacecolor='blue', label='Predicted')
                ax.set_title(s1.get() + " vs. " + s3.get())
                ax.set_xlabel(s1.get())
                ax.set_ylabel(s3.get())
                ax.legend()

                canvas = FigureCanvasTkAgg(figure, master=n2_root)
                canvas.draw()
                canvas.get_tk_widget().pack()

                _ = sm.OLS(y, X).fit()
                Label(n2_root, text='R^2 = {}'.format(_.rsquared_adj), bg='white', fg='black', font="Times 15").place(relx=0.4, rely=0.64)
            else:
                X, y = df[[s1.get().upper(), s2.get().upper()]], df[[s3.get().upper()]]
                ols = sm.OLS(y, X).fit()

                x_surf, y_surf = np.meshgrid(np.linspace(df[[s1.get().upper()]].min(),
                                                         df[[s1.get().upper()]].max(),
                                                         100),
                                             np.linspace(df[[s2.get().upper()]].min(),
                                                         df[[s2.get().upper()]].max(),
                                                         100))
                onlyx = pd.DataFrame({s1.get().upper(): x_surf.ravel(), s2.get().upper(): y_surf.ravel()})
                fittedy = ols.predict(exog=onlyx)

                fittedy = np.array(fittedy)

                ax = figure.add_subplot(111, projection='3d')
                ax.scatter(df[[s1.get().upper()]], df[[s2.get().upper()]],
                           df[[s3.get().upper()]], c='red', marker='o', alpha=0.5)
                ax.plot_surface(x_surf, y_surf, fittedy.reshape(x_surf.shape), color='white', alpha=0.3)
                ax.set_xlabel(s1.get())
                ax.set_ylabel(s2.get())
                ax.set_zlabel(s3.get())
                plt.show()



        Button(n_root, text="Compute Regression Model", command=regressionals).place(relx=0.38, rely=0.66, height=100, width=260)


    s0 = StringVar()
    s0.set("Select Mode")

    s1 = StringVar()
    s1.set("Select Predictor")

    s2 = StringVar()
    s2.set("Select Predictor")

    s3 = StringVar()
    s3.set("Select Dependent")

    m0 = OptionMenu(n_root, s0, "Simple Linear Regression", "Multiple Linear Regression", command=regression_mode)
    m0.place(relx=0.38, rely=0.4, height=40, width=260)

    m1 = OptionMenu(n_root, s1, "Age", "Gender", "Total Donations",
                    "First Year", "Num Fundraisers Attended")

    m2 = OptionMenu(n_root, s2, "Age", "Gender", "Total Donations",
                    "First Year", "Num Fundraisers Attended")

    m3 = OptionMenu(n_root, s3, "Age", "Gender", "Total Donations",
                    "First Year", "Num Fundraisers Attended", "Weights")


def manage_data():
    n_root = Toplevel()
    n_root.title("Data Manager")
    n_root.geometry("500x300")
    n_root.config(bg="gray")
    root.withdraw()

    def go_back():
        n_root.withdraw()
        root.deiconify()
    n_root.protocol("WM_DELETE_WINDOW", go_back)

    def display_data():
        n2_root = Toplevel()
        n2_root.resizable(False, False)
        n2_root.title("Insert Entry")
        n2_root.geometry("1629x700")
        n2_root.config(bg="gray")
        n_root.withdraw()

        my_tree = ttk.Treeview(n2_root)

        my_tree['columns'] = ("NAME", "EMAIL", "PHONE NUMBER", "AGE", "GENDER (0M/1F)",

                               "TOTAL DONATIONS", "LARGEST DONATION", "FIRST YEAR",
                               "NUM FUNDRAISERS ATTENDED")

        columns = ["NAME", "EMAIL", "PHONE NUMBER", "AGE", "GENDER (0M/1F)",
                   "TOTAL DONATIONS", "LARGEST DONATION", "FIRST YEAR",
                   "NUM FUNDRAISERS ATTENDED"]

        #initialize the columns
        my_tree.column("#0", width=0, minwidth=0)
        for label in columns:
            my_tree.column(label, anchor=W, width=100)

        #create headings
        my_tree.heading("#0", text="", anchor=W)
        for label in columns:
            my_tree.heading(label, text=label, anchor=W)

        #add data
        conn = sqlite3.connect("attendees.db")
        c = conn.cursor()
        data = c.execute("SELECT * FROM attendee_list")
        i = 0
        for attendee in data:
            info = []
            for j in range(len(attendee)):
                info.append(attendee[j])
            my_tree.insert(parent='', index='end', text='{}'.format(i),
                           iid='{}'.format(i), values=tuple(info))
            i += 1

        #scrollbars
        scrollbarx = Scrollbar(n2_root, orient=HORIZONTAL)
        scrollbary = Scrollbar(n2_root, orient=VERTICAL)

        my_tree.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)

        scrollbarx.configure(command=my_tree.xview)
        scrollbary.configure(command=my_tree.yview)
        scrollbarx.place(relx=0.0403, rely=0.499, width=1497.6, height=22)
        scrollbary.place(relx=0.961, rely=0.0709, width=22, height=297.6)

        my_tree.place(relx=0.04, rely=0.07, width=1500, height=300)

        def go_back():
            n2_root.withdraw()
            n_root.deiconify()
        n2_root.protocol("WM_DELETE_WINDOW", go_back)

    def insert():
        n2_root = Toplevel()
        n2_root.resizable(False, False)
        n2_root.title("Insert Entry")
        n2_root.geometry("1300x200")
        n2_root.config(bg="gray")
        n_root.withdraw()

        def go_back():
            n2_root.withdraw()
            n_root.deiconify()
        n2_root.protocol("WM_DELETE_WINDOW", go_back)

        Label(n2_root, text="NAME", bg='gray').place(relx=0.093, rely=0.05)
        name = StringVar()
        name_entry = Entry(n2_root, textvariable=name)
        name_entry.place(relx=0.06, rely=0.15)

        Label(n2_root, text="EMAIL", bg='gray').place(relx=0.224, rely=0.05)
        email = StringVar()
        email_entry = Entry(n2_root, textvariable=email)
        email_entry.place(relx=0.19, rely=0.15)

        Label(n2_root, text="PHONE-NUMBER", bg='gray').place(relx=0.332, rely=0.05)
        pn = StringVar()
        pn_entry = Entry(n2_root, textvariable=pn)
        pn_entry.place(relx=0.32, rely=0.15)

        Label(n2_root, text="AGE", bg='gray').place(relx=0.488, rely=0.05)
        age = StringVar()
        age_entry = Entry(n2_root, textvariable=age)
        age_entry.place(relx=0.45, rely=0.15)

        Label(n2_root, text="GENDER (M/F)", bg='gray').place(relx=0.595, rely=0.05)
        gender = StringVar()
        gen_entry = Entry(n2_root, textvariable=gender)
        gen_entry.place(relx=0.58, rely=0.15)

        Label(n2_root, text="DONATION", bg='gray').place(relx=0.732, rely=0.05)
        donation = StringVar()
        donation_entry = Entry(n2_root, textvariable=donation)
        donation_entry.place(relx=0.71, rely=0.15)

        Label(n2_root, text="YEAR", bg='gray').place(relx=0.876, rely=0.05)
        year = StringVar()
        year_entry = Entry(n2_root, textvariable=year)
        year_entry.place(relx=0.84, rely=0.15)

        entries = [name_entry, email_entry, pn_entry, age_entry, gen_entry,
                   donation_entry, year_entry]

        def insert_data():
            g = gender.get()
            if g == 'M':
                g = 0
            else:
                g = 1

            conn = sqlite3.connect("attendees.db")
            c = conn.cursor()

            #check if person already in database
            p = c.execute("SELECT EXISTS (SELECT * FROM attendee_list WHERE NAME = '{}')".format(name.get()))
            if p.fetchall()[0][0] == 1: # FOUND
                #update email
                c.execute("UPDATE attendee_list SET EMAIL = '{}' WHERE NAME = '{}'".format(email.get(), name.get()))
                #update phone number
                c.execute("UPDATE attendee_list SET PHONE_NUMBER = '{}' WHERE NAME = '{}'".format(pn.get(), name.get()))
                #update age
                c.execute("UPDATE attendee_list SET AGE = {} WHERE NAME = '{}'".format(int(age.get()), name.get()))
                #update total donations
                c.execute("UPDATE attendee_list SET TOTAL_DONATIONS = TOTAL_DONATIONS + {} WHERE NAME = '{}'".format(int(donation.get()), name.get()))
                #update largest donation
                largest_d = c.execute("SELECT LARGEST_DONATION FROM attendee_list WHERE NAME = '{}'".format(name.get()))
                if int(donation.get()) > int(largest_d.fetchall()[0][0]):
                    c.execute("UPDATE attendee_list SET LARGEST_DONATION = {} WHERE NAME = '{}'".format(int(donation.get()), name.get()))
                else:
                    pass
                #update number of fundraisers attended
                c.execute("UPDATE attendee_list SET NUM_FUNDRAISERS_ATTENDED = NUM_FUNDRAISERS_ATTENDED + 1 WHERE NAME = '{}'".format(name.get()))


            else: # NOT FOUND
                c.execute("INSERT INTO attendee_list VALUES (:NAME, :EMAIL, :PHONE_NUMBER, :AGE, :GENDER, :TOTAL_DONATIONS, :LARGEST_DONATION, :FIRST_YEAR, :NUM_FUNDRAISERS_ATTENDED)",
                          {
                              'NAME': name.get(),
                              'EMAIL': email.get(),
                              'PHONE_NUMBER': pn.get(),
                              'AGE': int(age.get()),
                              'GENDER': g,
                              'TOTAL_DONATIONS': int(donation.get()),
                              'LARGEST_DONATION': int(donation.get()),
                              'FIRST_YEAR': int(year.get()),
                              'NUM_FUNDRAISERS_ATTENDED': 1
                          })
                pass


            c.close()
            conn.commit()
            conn.close()

            for entry in entries:
                entry.delete(0, END)

        Button(n2_root, text="Insert", font="Times 15", command=insert_data, width=10, height=2).place(relx=0.453, rely=0.45)

    insert_btn = Button(n_root, text="Insert Entry", font="Times 25", command=insert)
    insert_btn.place(relx=0.05, rely=0.05, relheight=0.9, relwidth=0.45)

    display_btn = Button(n_root, text="Display Data", font="Times 25", command=display_data)
    display_btn.place(relx=0.5, rely=0.05, relheight=0.9, relwidth=0.45)


#---------------------------- HOME PAGE BUTTONS AND LABELS ---------------------

c = Canvas(root, bg='black')
c.place(relwidth=1, relheight=0.2)

header = Label(c, text="Attendance Data Prediction", fg="white", bg="black", font="{Comic Sans MS} 38")
header.config(height=100, width=100)
header.pack()

load_model_btn = Button(root, text="Load Prediction Model", font="Times 25", command=load_model, borderwidth=2)
load_model_btn.place(relx=0.05, rely=0.31, relwidth=0.9, relheight=0.3)

insert_data = Button(root, text="Manage Data", font="Times 25", command=manage_data, borderwidth=2)
insert_data.place(relx=0.05, rely=0.62, relwidth=0.9, relheight=0.3)


root.resizable(False, False)
root.mainloop()


