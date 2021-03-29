def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
sql = '''
SELECT *
FROM properties_2017
JOIN (SELECT parcelid, max(logerror) as logerror, max(transactiondate) as transactiondate 
FROM predictions_2017 group by parcelid) as pred_17 using(parcelid) 
LEFT JOIN airconditioningtype using(airconditioningtypeid) 
LEFT JOIN architecturalstyletype using(architecturalstyletypeid) 
LEFT JOIN buildingclasstype using(buildingclasstypeid) 
LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid) 
LEFT JOIN storytype using(storytypeid) 
LEFT JOIN typeconstructiontype using(typeconstructiontypeid) 
WHERE year(transactiondate) = 2017;
'''


## df = pd.read_sql(sql, get_connection('zillow'))

def missing_value_table(df):
    '''
    This function will take in a dataframe and will return a report detailing missing values in
    the dataframes various columns
    '''
    
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)

    return mz_table


def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


