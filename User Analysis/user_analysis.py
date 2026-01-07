import pandas as pd

def analyze_dates():
    
    df = pd.read_csv("C:/Users/Bo$$/Downloads/dat1.csv",encoding ="latin1")

    df.columns = df.columns.str.strip().str.replace('"', '')

    df['DATE_CREATED'] = pd.to_datetime(df['DATE_CREATED'], errors='coerce')
    df['LAST_ACTIVE'] = pd.to_datetime(df['LAST_ACTIVE'], errors='coerce')

    df = df.dropna(subset=['DATE_CREATED', 'LAST_ACTIVE'])

    df['CREATED_YEAR'] = df['DATE_CREATED'].dt.year
    df['CREATED_MONTH'] = df['DATE_CREATED'].dt.strftime('%B')
    df['CREATED_YEAR_MONTH'] = df['DATE_CREATED'].dt.strftime('%Y-%m')

    df['ACTIVE_YEAR'] = df['LAST_ACTIVE'].dt.year
    df['ACTIVE_MONTH'] = df['LAST_ACTIVE'].dt.strftime('%B')
    df['ACTIVE_YEAR_MONTH'] = df['LAST_ACTIVE'].dt.strftime('%Y-%m')

   
    accounts_per_year = df.groupby('CREATED_YEAR').size().reset_index(name='Total_Accounts')
    print("Total Accounts Created Per Year:")
    print(accounts_per_year.to_string(index=False))

    accounts_per_month = df.groupby(['CREATED_YEAR', 'CREATED_MONTH']).size().reset_index(name='Total_Accounts')
    most_accounts_month = accounts_per_month.loc[accounts_per_month.groupby('CREATED_YEAR')['Total_Accounts'].idxmax()]
    print("\nMonth with Most Accounts Created Per Year:")
    print(most_accounts_month.to_string(index=False))

    active_per_year = df.groupby('ACTIVE_YEAR').size().reset_index(name='Active_Accounts')
    total_accounts = len(df)
    active_per_year['Activity_Rate'] = active_per_year['Active_Accounts'] / total_accounts
    print("\nRate of Activity and Sum of Active Accounts Per Year:")
    print(active_per_year.to_string(index=False))

    active_per_month = df.groupby(['ACTIVE_YEAR', 'ACTIVE_MONTH']).size().reset_index(name='Active_Accounts')
    print("\nActive Accounts Per Month Per Year:")
    print(active_per_month.to_string(index=False))

    most_active_year = active_per_year.loc[active_per_year['Active_Accounts'].idxmax()]
    print(f"\nMost Active Year: {int(most_active_year['ACTIVE_YEAR'])} with {int(most_active_year['Active_Accounts'])} active accounts")

    most_active_month = active_per_month.loc[active_per_month['Active_Accounts'].idxmax()]
    print(f"Most Active Month: {most_active_month['ACTIVE_MONTH']} {int(most_active_month['ACTIVE_YEAR'])} with {int(most_active_month['Active_Accounts'])} active accounts")

if __name__ == "__main__":
    analyze_dates()