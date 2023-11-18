from tkinter import ttk, filedialog, messagebox
import customtkinter as tk
import pandas as pd
tk.set_appearance_mode('dark')
tk.set_default_color_theme('blue')

# init window frame
window = tk.CTk()
df = pd.DataFrame
# config window frame
window.minsize(width=1280,height=720)
window.title('Predict MaketPlace')


frame_input = tk.CTkFrame(master=window)
# Show data_name choise
input_name = tk.CTkLabel(master=frame_input,height=20,width=200,text='')
input_name.pack()
def file_input():
    global df
    name = filedialog.askopenfilename(title='Select csv file',filetypes=[('comma-separated values', '*.csv')])
    input_name.configure(text=name)

    # Try to open file
    try:
        df = pd.read_csv(name)
    except Exception as e:
        messagebox.showerror("Something wrong ! ",str(e))
    # Get header
    table['column'] = list(df.columns)
    table['show'] = 'headings'
    # Show header
    for col in table['column']:
        table.heading(col,text=col)
        table.column(col,width=100,stretch= True)
    # Show data
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        table.insert("","end", values=row)

input_button = tk.CTkButton(master=window,text = 'Open file',
                            command=lambda :file_input(),
                            width=100,height=25,
                            corner_radius=5,
                            fg_color='#336fd6',hover_color='#0a1f42')
input_button.pack()

# Data Table

table = ttk.Treeview(frame_input)
# vertical scrollbar

vsb = ttk.Scrollbar(frame_input, orient="vertical", command=table.yview)
vsb.pack(side='right', fill='y')

# Table style
table_style = ttk.Style()
table_style.theme_use('default')


# - for row below heading
table_style.configure("Treeview",
                      background = '#374a6b',
                      foreground= 'black',
                      fieldbackground='#374a6b')

# - for heading
table_style.configure("Treeview.Heading",
                      background='#162236',
                      foreground='black',
                      fieldbackground='#162236')
table.configure(yscrollcommand=vsb.set)
# #162236
table.pack(expand=True)
frame_input.pack()


window.mainloop()