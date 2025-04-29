import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts, JsCode
import os
import pyarrow.parquet as pq
from pathlib import Path

st.set_page_config(page_title="Waterfall Chart", page_icon=":bar_chart:", layout="wide")

# Define file paths
CSV_PATH = Path('scanntech.csv')
PARQUET_PATH = Path('scanntech.parquet')

def convert_csv_to_parquet(csv_path, parquet_path):
    """Reads a CSV file and saves it as a Parquet file."""
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        df.to_parquet(parquet_path, index=False)
        st.toast(f"Converted {csv_path.name} to {parquet_path.name}")
    except Exception as e:
        st.error(f"Error converting CSV to Parquet: {e}")

@st.cache_data
def load_data():
    """Loads data from Parquet, converting from CSV if necessary."""
    # Check if Parquet needs update
    needs_update = False
    if not PARQUET_PATH.exists():
        needs_update = True
        st.toast(f"{PARQUET_PATH.name} not found. Converting from CSV...")
    elif CSV_PATH.exists() and CSV_PATH.stat().st_mtime > PARQUET_PATH.stat().st_mtime:
        needs_update = True
        st.toast(f"{CSV_PATH.name} is newer than {PARQUET_PATH.name}. Re-converting...")

    if needs_update:
        if CSV_PATH.exists():
            convert_csv_to_parquet(CSV_PATH, PARQUET_PATH)
        else:
            st.error(f"Error: {CSV_PATH.name} not found. Cannot create Parquet file.")
            return pd.DataFrame() # Return empty DataFrame if CSV is missing

    # Load from Parquet
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        st.error(f"Error loading Parquet file {PARQUET_PATH.name}: {e}")
        # Fallback or further error handling could go here
        # For now, try loading CSV as a last resort if Parquet fails after conversion attempt
        if CSV_PATH.exists():
             st.warning(f"Falling back to loading {CSV_PATH.name} directly.")
             try:
                 df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
             except Exception as csv_e:
                 st.error(f"Error loading CSV file {CSV_PATH.name}: {csv_e}")
                 return pd.DataFrame() # Return empty if both fail
        else:
            return pd.DataFrame() # Return empty if Parquet fails and CSV missing

    # --- Data Transformations (applied after loading) ---
    df['03. Produto Sellout [Marca]'] = df['03. Produto Sellout [Marca]'].replace('BARE TUTTI FRUTTI', 'BARE GUARANA')
    df['Marca Ajustada'] = df['Marca Ajustada'].replace('BARE TUTTI FRUTTI', 'BARE GUARANA')
    df['Marca TT'] = df['Marca TT'].replace('BARE TUTTI FRUTTI', 'BARE GUARANA')
    df['nome_produto'] = df['03. Produto Sellout [Marca]'] + ' ' + df['03. Produto Sellout [Tamanho Embalagem]'].astype(str)
    # Add MS/SS classification
    df['MS/SS'] = df['03. Produto Sellout [Tamanho Embalagem]'].apply(lambda x: 'MS' if x >= 1000 else 'SS')
    return df

@st.cache_data
def compute_waterfall(df, variavel='Marca TT', top_n=7):
    vol = '[Volume (Hl) Sellout]'
    
    # Filter Jan-Apr data
    months = df['01. Períodos[Mês Nielsen]'].isin([1,2,3,4])
    q = df[months]
    
    q24 = q[q['01. Períodos[Ano Nielsen]'] == 2024]
    q25 = q[q['01. Períodos[Ano Nielsen]'] == 2025]
    tot24 = q24[vol].sum()
    tot25 = q25[vol].sum()
    ambev24 = q24[q24['Fabricante Ajustado']=='AMBEV'][vol].sum()
    ambev25 = q25[q25['Fabricante Ajustado']=='AMBEV'][vol].sum()
    share_start = ambev24/tot24*100 if tot24 else 0
    share_end = ambev25/tot25*100 if tot25 else 0
    expected_impact = share_end - share_start

    # Calculate brand contributions to share change
    brand = variavel
    brands24 = q24.groupby(brand)[vol].sum()
    brands25 = q25.groupby(brand)[vol].sum()
    
    # Get set of Ambev brands for the selected variable
    ambev_brands = set(df[df['Fabricante Ajustado'] == 'AMBEV'][variavel].unique())

    # Calculate share impact for each brand using approximation
    share_impacts = {}
    all_brands = set(brands24.index) | set(brands25.index)
    
    if tot24 > 0: # Avoid division by zero
        for b in all_brands:
            vol24 = brands24.get(b, 0)
            vol25 = brands25.get(b, 0)
            # Calculate absolute share change for the brand
            share24 = (vol24 / tot24 * 100) if tot24 > 0 else 0
            share25 = (vol25 / tot25 * 100) if tot25 > 0 else 0
            absolute_share_change = share25 - share24

            # Determine contribution based on whether it's an Ambev brand
            if b in ambev_brands:
                contribution_b = absolute_share_change # Direct share change for Ambev brands
            else:
                contribution_b = -absolute_share_change # Inverse share change for competitors

            share_impacts[b] = contribution_b
    else: # Handle case where initial total volume is zero or final total volume is zero
         for b in all_brands:
             # If tot24 is 0 but tot25 is not, the change is effectively the new share
             if tot24 == 0 and tot25 > 0:
                 share25 = (brands25.get(b, 0) / tot25 * 100)
                 if b in ambev_brands:
                     share_impacts[b] = share25
                 else:
                     share_impacts[b] = -share25
             # If tot25 is 0 but tot24 is not, the change is the negative of the old share
             elif tot25 == 0 and tot24 > 0:
                 share24 = (brands24.get(b, 0) / tot24 * 100)
                 if b in ambev_brands:
                     share_impacts[b] = -share24
                 else:
                     share_impacts[b] = share24 # Competitor losing share benefits Ambev
             else: # Both are zero or some other edge case
                 share_impacts[b] = 0

    # Sort impacts and split into top gains/losses
    sorted_impacts = sorted(share_impacts.items(), key=lambda x: x[1], reverse=True) # Sort by value
    pos_impacts = [(b,v) for b,v in sorted_impacts if v > 0][:top_n]
    neg_impacts = sorted([(b,v) for b,v in sorted_impacts if v < 0], key=lambda x: x[1])[:top_n] # Sort negatives ascending

    # Calculate the sum of the top N impacts shown
    top_n_sum = sum(v for b,v in pos_impacts) + sum(v for b,v in neg_impacts)
    
    # Calculate the 'Other' value needed to reconcile the total change
    other_sum_calculated = sum(v for b,v in share_impacts.items() 
                               if b not in dict(pos_impacts + neg_impacts))
    other_sum_reconciled = expected_impact - top_n_sum

    # Build waterfall rows
    rows = [{'Category': 'Start AMBEV Share', 'Value': share_start, 'step': 'total'}]
    
    # Add positive impacts
    for b, v in pos_impacts:
        rows.append({'Category': b, 'Value': v, 'step': 'increase'})
    
    # Add negative impacts
    for b, v in neg_impacts:
        rows.append({'Category': b, 'Value': v, 'step': 'decrease'})
    
    # Add reconciled other impact
    if abs(other_sum_reconciled) > 1e-6: # Use a small threshold to avoid adding near-zero 'Other' bars
        rows.append({'Category': 'Other', 'Value': other_sum_reconciled, 'step': 'other'}) # Assign 'other' step
        # Optional: Add a warning if the calculated 'Other' differs significantly from the reconciled one
       
    # Add End Share bar - Value should be the target end share, not calculated running total
    rows.append({'Category': 'End AMBEV Share', 'Value': share_end, 'step': 'total'})
    
    # Create DataFrame and calculate running totals for bar positions
    df_wf = pd.DataFrame(rows)
    df_wf['running_total'] = df_wf['Value'].cumsum()
    df_wf['bottom'] = df_wf['running_total'] - df_wf['Value']
    df_wf['top'] = df_wf['running_total']

    # Adjust 'bottom' and 'top' for waterfall display logic
    # Start bar: bottom=0, top=start_value
    # Increase bars: bottom=previous_running_total, top=current_running_total
    # Decrease bars: bottom=current_running_total, top=previous_running_total
    # End bar: bottom=0, top=end_value
    
    new_bottom = [0.0] # Start bar bottom is 0
    new_top = [df_wf.loc[0, 'Value']] # Start bar top is start_value
    
    for i in range(1, len(df_wf) - 1): # Iterate through impact bars
        prev_running_total = df_wf.loc[i-1, 'running_total']
        current_running_total = df_wf.loc[i, 'running_total']
        step_type = df_wf.loc[i, 'step']
        
        if step_type == 'increase':
            new_bottom.append(prev_running_total)
            new_top.append(current_running_total)
        elif step_type == 'decrease':
             new_bottom.append(current_running_total)
             new_top.append(prev_running_total)
        else: # Handle 'other' based on its sign (step assigned earlier)
             # Note: This logic block for 'other' might need review if 'other' step is used
             # For now, assume 'other' behaves like increase/decrease based on value sign
             # If 'other' has specific positioning rules, adjust here.
             # The current code handles 'other' step type but positions based on value sign.
             if df_wf.loc[i, 'Value'] >= 0: # Treat positive 'other' as increase
                 new_bottom.append(prev_running_total)
                 new_top.append(current_running_total)
             else: # Treat negative 'other' as decrease
                 new_bottom.append(current_running_total)
                 new_top.append(prev_running_total)

    # End bar
    new_bottom.append(0.0)
    new_top.append(df_wf.loc[len(df_wf)-1, 'Value']) # End bar top is end_value

    df_wf['bottom'] = new_bottom
    df_wf['top'] = new_top
    
    # The explicit override for start/end might be redundant now, but keep for clarity
    df_wf.loc[df_wf['Category'] == 'Start AMBEV Share', ['bottom', 'top']] = [0, share_start]
    df_wf.loc[df_wf['Category'] == 'End AMBEV Share', ['bottom', 'top']] = [0, share_end]

    return df_wf

# GLOBAL CONSTANTS FOR MONTH LABELS
month_abbr = ['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez']
month_cols = [f"{month_abbr[m-1]}/{str(y)[-2:]}" for y in [2024,2025] for m in range(1,13)]

@st.cache_data
def compute_price_index(filtered_df, variavel, rev_col, vol_col, all_df):
    """Vectorized monthly and Q1 price index & volumes."""
    if filtered_df.empty:
        return pd.DataFrame()
    # monthly price %
    df2 = (
        filtered_df
          .groupby([variavel,'01. Períodos[Ano Nielsen]','01. Períodos[Mês Nielsen]'])
          .agg({rev_col:'sum', vol_col:'sum'})
          .reset_index()
          .assign(price=lambda d: d[rev_col] / d[vol_col])
          .merge(
              all_df.groupby(['01. Períodos[Ano Nielsen]','01. Períodos[Mês Nielsen]'])
                    .agg({rev_col:'sum', vol_col:'sum'})
                    .assign(total_price=lambda d: d[rev_col]/d[vol_col])
                    .reset_index()[['01. Períodos[Ano Nielsen]','01. Períodos[Mês Nielsen]','total_price']],
              on=['01. Períodos[Ano Nielsen]','01. Períodos[Mês Nielsen]'], how='left'
          )
          .assign(pct=lambda d: d['price'] / d['total_price'] * 100)
    )
    tbl = df2.pivot(index=variavel,columns=['01. Períodos[Ano Nielsen]','01. Períodos[Mês Nielsen]'],values='pct')
    tbl.columns = [f"{month_abbr[m-1]}/{str(y)[-2:]}" for y,m in tbl.columns]
    # Q1 metrics
    q1 = (
        filtered_df[filtered_df['01. Períodos[Trimestre]']=='1 Tri']
          .groupby([variavel,'01. Períodos[Ano Nielsen]'])
          .agg({rev_col:'sum', vol_col:'sum'})
          .reset_index()
          .assign(price=lambda d: d[rev_col] / d[vol_col])
          .merge(
              all_df[all_df['01. Períodos[Trimestre]']=='1 Tri']
                .groupby('01. Períodos[Ano Nielsen]')
                .agg({rev_col:'sum', vol_col:'sum'})
                .assign(total_price=lambda d: d[rev_col]/d[vol_col])
                .reset_index()[['01. Períodos[Ano Nielsen]','total_price']],
              on='01. Períodos[Ano Nielsen]', how='left'
          )
    )
    q1['pct']=q1['price']/q1['total_price']*100
    qp=q1.pivot(index=variavel,columns='01. Períodos[Ano Nielsen]',values='pct')
    # pivot volumes using actual volume column name
    qv=q1.pivot(index=variavel,columns='01. Períodos[Ano Nielsen]',values=vol_col)
    tbl['2024 Q1']=qp.get(2024);
    tbl['2025 Q1']=qp.get(2025)
    tbl['2024 Q1 Vol']=qv.get(2024);
    tbl['2025 Q1 Vol']=qv.get(2025)
    tbl['Vol % Var']=((tbl['2025 Q1 Vol']-tbl['2024 Q1 Vol'])/tbl['2024 Q1 Vol']*100).round(1)
    tbl['Vol % Var']=tbl['Vol % Var'].fillna(0).astype(str)+'%'
    # prune and order
    keep=[c for c in month_cols+['2024 Q1','2025 Q1','2024 Q1 Vol','2025 Q1 Vol','Vol % Var'] if c in tbl]
    return tbl.loc[:,keep].sort_values('2025 Q1 Vol',ascending=False)

# Main app layout and filters
# Load data
all_df = load_data()
# Sidebar filters
st.sidebar.header('Filters')
variavel = st.sidebar.selectbox('Select Variable', ['Marca TT', 'Fabricante Ajustado', '03. Produto Sellout [Tamanho Embalagem]', 'nome_produto', 'MS/SS'], index=0)
ufs = sorted(all_df['04. Cliente[UF]'].unique())
selected_ufs = st.sidebar.selectbox('Select UF', ufs, index=0)
canals = sorted(all_df['04. Cliente[Canal]'].unique())
selected_canals = st.sidebar.multiselect('Select Canal', canals, default=canals)
manus = sorted(all_df['Fabricante Ajustado'].unique())
selected_manus = st.sidebar.multiselect('Select Manufacturer', manus, default=manus)
msss = sorted(all_df['MS/SS'].unique())
selected_msss = st.sidebar.multiselect('Select MS/SS', msss, default=msss)

# Apply all filters
filtered_df = all_df[
    (all_df['04. Cliente[UF]'] == selected_ufs) &
    all_df['04. Cliente[Canal]'].isin(selected_canals) &
    all_df['Fabricante Ajustado'].isin(selected_manus) &
    all_df['MS/SS'].isin(selected_msss)
]
# Compute waterfall
wf = compute_waterfall(filtered_df, variavel)

# Render ECharts waterfall chart
st.title('Waterfall: Q1 2024 → Q1 2025 Market Share')
categories = wf['Category'].tolist()
if categories:
    categories[0] = 'Q1 24'
    categories[-1] = 'Q1 25'
# The 'bottom' and 'top' columns now correctly represent the start and end points for each bar segment.
# The ECharts configuration needs to use these directly.
# We need placeholder bars starting at 'bottom' and value bars representing the magnitude |top - bottom|.

# Prepare data for ECharts series
placeholders = [round(b, 2) for b in wf['bottom'].tolist()]
magnitudes = [round(abs(t - b), 2) for b, t in zip(wf['bottom'].tolist(), wf['top'].tolist())]

# Assign colors based on step type
color_map = {'total':'#5470C6','increase':'#91CC75','decrease':'#EE6666', 'other': '#808080'} # Added grey for 'other'
bar_colors = []
for step in wf['step'].tolist(): # No need for value anymore
    if step == 'total':
        bar_colors.append(color_map['total'])
    elif step == 'increase':
         bar_colors.append(color_map['increase'])
    elif step == 'decrease':
         bar_colors.append(color_map['decrease'])
    elif step == 'other':
         bar_colors.append(color_map['other']) # Assign grey for 'other' step
    else: # Fallback (shouldn't happen with current logic)
        bar_colors.append('#CCCCCC') # Default fallback color


option = {
    'title': {
        'text': 'Waterfall Chart',
        'subtext': 'Q1 24 vs Q1 25'
    },
    'tooltip': {
        'trigger': 'axis',
        'axisPointer': {'type': 'shadow'}
    },
    'grid': {
        'left': '3%', 'right': '4%', 'bottom': '3%', 'containLabel': True
    },
    'xAxis': {
        'type': 'category',
        'splitLine': {'show': False},
        'axisLabel': {'interval': 0, 'rotate': 90},
        'data': categories
    },
    'yAxis': {'type': 'value'},
    'series': [
        {
            'name': 'Placeholder', 'type': 'bar', 'stack': 'Total',
            'itemStyle': {'borderColor': 'transparent', 'color': 'transparent'},
            'emphasis': {'itemStyle': {'borderColor': 'transparent', 'color': 'transparent'}},
            'data': placeholders # Use the calculated bottom values as placeholders
        },
        {
            'name': 'Value', 'type': 'bar', 'stack': 'Total',
            'label': {'show': True, 'position': 'top', 'formatter': JsCode("function(params){ if(params.value > 0) {return params.value.toFixed(2);} else {return '';} }").js_code}, # Show label only for positive magnitude (actual bars)
            'data': [
                {'value': mag, 'itemStyle': {'color': col}} 
                for mag, col in zip(magnitudes, bar_colors) # Use calculated magnitudes and colors
            ]
        }
    ]
}
st_echarts(options=option, height='600px')

# Add Monthly Share Line Chart
def compute_monthly_share(df):
    vol = '[Volume (Hl) Sellout]'
    monthly = df.groupby(['01. Períodos[Ano Nielsen]', '01. Períodos[Mês Nielsen]'])[vol].sum().reset_index()
    ambev = df[df['Fabricante Ajustado']=='AMBEV'].groupby(['01. Períodos[Ano Nielsen]', '01. Períodos[Mês Nielsen]'])[vol].sum().reset_index()
    merged = monthly.merge(ambev, on=['01. Períodos[Ano Nielsen]', '01. Períodos[Mês Nielsen]'], suffixes=('_total', '_ambev'))
    merged['share'] = (merged[f'{vol}_ambev'] / merged[f'{vol}_total'] * 100).round(2)
    return merged

monthly_share = compute_monthly_share(filtered_df)
share_2024 = monthly_share[monthly_share['01. Períodos[Ano Nielsen]'] == 2024]['share'].tolist()
share_2025 = monthly_share[monthly_share['01. Períodos[Ano Nielsen]'] == 2025]['share'].tolist()

# Calculate Q1 average using total volumes instead of monthly averages
vol = '[Volume (Hl) Sellout]'
q1_2024_data = filtered_df[(filtered_df['01. Períodos[Ano Nielsen]'] == 2024) & 
                          (filtered_df['01. Períodos[Mês Nielsen]'].isin([1,2,3,4]))]
q1_2025_data = filtered_df[(filtered_df['01. Períodos[Ano Nielsen]'] == 2025) & 
                          (filtered_df['01. Períodos[Mês Nielsen]'].isin([1,2,3,4]))]

q1_2024 = round((q1_2024_data[q1_2024_data['Fabricante Ajustado']=='AMBEV'][vol].sum() / 
                 q1_2024_data[vol].sum() * 100) if not q1_2024_data.empty else 0, 2)
q1_2025 = round((q1_2025_data[q1_2025_data['Fabricante Ajustado']=='AMBEV'][vol].sum() / 
                 q1_2025_data[vol].sum() * 100) if not q1_2025_data.empty else 0, 2)

line_option = {
    'title': {
        'text': 'Market Share Evolution',
        'subtext': 'Q1 Average & Monthly Trend'
    },
    'tooltip': {
        'trigger': 'axis',
        'axisPointer': {'type': 'shadow'},
        
    },
    'legend': {
        'data': ['Q1 2024', 'Q1 2025', '2024 Monthly', '2025 Monthly'],
        'selected': {
            'Q1 2024': True,
            'Q1 2025': True,
            '2024 Monthly': True,
            '2025 Monthly': True
        }
    },
    'xAxis': [{
        'type': 'category',
        'data': ['Q1 Average'],
        'position': 'bottom',
        'axisLabel': {'rotate': 0},
        'splitLine': {'show': False},
        'gridIndex': 0,
        'left': '5%',
        'width': '10%'
    }, {
        'type': 'category',
        'data': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'gridIndex': 1,
        'position': 'bottom',
        'left': '20%',
        'width': '75%'
    }],
    'yAxis': [
        {
            'type': 'value',
            'gridIndex': 0,
            'axisLabel': {'formatter': '{value}%'}
        },
        {
            'type': 'value',
            'gridIndex': 1,
            'axisLabel': {'formatter': '{value}%'}
        }
    ],
    'grid': [
        {'left': '5%', 'width': '10%', 'containLabel': True},
        {'left': '20%', 'width': '75%', 'containLabel': True}
    ],
    'series': [
        {
            'name': 'Q1 2024',
            'type': 'bar',
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'data': [q1_2024],
            'color': '#5470C6',
            'label': {'show': True, 'formatter': '{c}%', 'position': 'top'}
        },
        {
            'name': 'Q1 2025',
            'type': 'bar',
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'data': [q1_2025],
            'color': '#FF7070',
            'label': {'show': True, 'formatter': '{c}%', 'position': 'top'}
        },
        {
            'name': '2024 Monthly',
            'type': 'line',
            'smooth': True,
            'xAxisIndex': 1,
            'yAxisIndex': 1,
            'data': share_2024,
            'color': '#5470C6',
            'label': {'show': True, 'formatter': '{c}%', 'position': 'top'}
        },
        {
            'name': '2025 Monthly',
            'type': 'line',
            'smooth': True,
            'xAxisIndex': 1,
            'yAxisIndex': 1,
            'color': '#FF7070',
            'data': share_2025,
            'label': {'show': True, 'formatter': '{c}%', 'position': 'top'}
        }
    ]
}
st_echarts(options=line_option, height='400px')

# Price Index Table: Filtered Data vs Total (Q1 2024/2025)
st.subheader('Price Index')
rev_col = '[Faturamento/GMV (R$) Sellout]'
vol_col = '[Volume (Hl) Sellout]'
years = [2024, 2025]
table = pd.DataFrame(index=[f'Q1 {y}' for y in years], columns=['Filtered Price', 'Total Price', 'Price Index'])
all_df_q1 = all_df[all_df['01. Períodos[Trimestre]'] == '1 Tri']
for y in years:
    # Filtered data
    f_q1 = filtered_df[(filtered_df['01. Períodos[Trimestre]'] == '1 Tri') & (filtered_df['01. Períodos[Ano Nielsen]'] == y)]
    f_vol = f_q1[vol_col].sum()
    f_rev = f_q1[rev_col].sum()
    f_price = f_rev / f_vol if f_vol else None
    # Total data
    t_q1 = all_df_q1[all_df_q1['01. Períodos[Ano Nielsen]'] == y]
    t_vol = t_q1[vol_col].sum()
    t_rev = t_q1[rev_col].sum()
    t_price = t_rev / t_vol if t_vol else None
    # Index
    idx = f_price / t_price * 100 if f_price and t_price else None
    table.loc[f'Q1 {y}', 'Filtered Price'] = f'{f_price:.2f}' if f_price else ''
    table.loc[f'Q1 {y}', 'Total Price'] = f'{t_price:.2f}' if t_price else ''
    table.loc[f'Q1 {y}', 'Price Index'] = f'{idx:.1f}' if idx else ''
st.table(table)

# --- Price Index Table by Variable ---
st.subheader(f'Price Index by {variavel} (vs Total, by Month and Q1)')
tbl = compute_price_index(filtered_df, variavel, rev_col, vol_col, all_df)
# format display: percentify price-index, round volumes
display_tbl = tbl.copy()
# percent columns: monthly & Q1 indexes
for col in month_cols + ['2024 Q1', '2025 Q1']:
    if col in display_tbl:
        display_tbl[col] = display_tbl[col].round(1).astype(str) + '%'
# volume columns
for col in ['2024 Q1 Vol', '2025 Q1 Vol']:
    if col in display_tbl:
        display_tbl[col] = display_tbl[col].round(1)
# Vol % Var already has '%'
st.dataframe(display_tbl, height=2000)
