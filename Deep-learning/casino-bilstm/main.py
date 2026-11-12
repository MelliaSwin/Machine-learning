"""
main.py

What is done here:
    - Runs everything
    - Display Visuals

"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.patches import Patch
from train import train
from config import config_args

"""
Function: timeline_plot:
    - For individual customers to show their behaviors
"""
def timeline_plot(customer_id, df_model, save_path, title_prefix="Customer"):
    # Grab last 50 transactions for selected customer
    customer_data = df_model[df_model['AccountIdentifier'] == customer_id].tail(50).copy()
    customer_data['Step'] = range(len(customer_data))

    plt.figure(figsize=[14, 6])
    # Red = Declined Transactions, Green = Approved Transactions
    colors = customer_data['Is_Declined'].apply(
        lambda x: 'red'
        if x==1
        else 'green'
    )

    # Bar chart for Transaction Amounts
    axis_1 = plt.gca()
    axis_1.bar(customer_data['Step'],
               customer_data['TransactionAmount'],
               color=colors,
               alpha=0.5,
               label='Amount')
    axis_1.set_ylabel('Transaction Amount ($)')
    axis_1.set_xlabel('Transaction Sequence (Last 50)')
    # Layered ontop of bar chart, line plot
    axis_2 = axis_1.twinx()
    axis_2.plot(customer_data['Step'],
                customer_data['DeltaTime'],
               color='blue',
                marker='o',
                linewidth=2,
                label='Time Gap')
    axis_2.set_ylabel('Time Gap (Seconds)', color='blue')
    plt.title(f'{title_prefix}: {customer_id}')

    # Custom Legend:
    legend_elements = [Patch(facecolor='green', label='Approved'),
                       Patch(facecolor='red', label='Declined'),
                       plt.Line2D([0], [0], color='blue', marker='o', label='Time Gap')]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
Function: visuals:
    - Histogram of anomaly scores
    - Timeline plot for high-risk and normal customers
    - Comparison barplot of feature-level anomalies
"""
def visuals():
    args = config_args.parse_args()

    # Here, we train model
    model, results_df, df_model = train()

    # Save to directories
    base_dir = args.output_dir
    risk_dir = os.path.join(base_dir, 'Risk_Customers')
    normal_dir = os.path.join(base_dir, 'Normal_Customers')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(risk_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # Histogram
    threshold = results_df['Total_Anomaly_Score'].quantile(args.risk_quantile)

    plt.figure(figsize=[12, 6])
    plt.hist(results_df['Total_Anomaly_Score'],
             bins=50,
             color='skyblue',
             edgecolor='black',
             alpha=0.7)
    plt.axvline(threshold, color='red',
                linestyle='dashed',
                linewidth=2,
                label=f'Threshold({threshold:.4f})')
    plt.title('Distribution of Total_Anomaly_Score')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Customers')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'anomaly_histogram.png'))
    plt.close()

    # Identify Groups:
    # Grab top 5 customers at risk
    top_risk_customers = results_df.sort_values(by='Total_Anomaly_Score', ascending=False).head(5)

    # Grab top 5 customers that are normal (BUT they must have >50 transactions for fairness)
    customer_counts = df_model['AccountIdentifier'].value_counts()
    active_ids = customer_counts[customer_counts >= 50].index
    active_results = results_df[results_df['CustomerIdentifier'].isin(active_ids)]
    normal_customers = active_results.sort_values(by='Total_Anomaly_Score', ascending=True).head(5)

    # Plot both RISK and NORMAL
    for _, row in top_risk_customers.iterrows():
        uid = row['CustomerIdentifier']
        save_path = os.path.join(risk_dir, f"{uid}_timeline.png")
        timeline_plot(uid, df_model, save_path, title_prefix="HIGH RISK")

    for _, row in normal_customers.iterrows():
        uid = row['CustomerIdentifier']
        save_path = os.path.join(normal_dir, f"{uid}_timeline.png")
        timeline_plot(uid, df_model, save_path, title_prefix="NORMAL")

    # Comparison between RISK and NORMAL customers
    plot_data = pd.concat([top_risk_customers, normal_customers])
    plot_data['Group'] = ['High Risk'] * 5 + ['Active Normal'] * 5
    plot_melt = plot_data.melt(id_vars=['CustomerIdentifier', 'Group'],
                               value_vars=['Error_Time', 'Error_Amount', 'Error_Bank',
                                           'Error_Wager', 'Error_Cashout', 'Error_Decline'],
                               var_name='Error_Source', value_name='Magnitude')
    plt.figure(figsize=[14, 7])
    sns.barplot(data=plot_melt,
                x='CustomerIdentifier',
                y='Magnitude',
                hue='Error_Source',
                palette='tab10')
    plt.title('High Risk vs Normal Customers')
    plt.xticks(rotation=45)
    plt.ylabel('Reconstruction Error')
    plt.axvline(4.5,
                color='black',
                linestyle='--',
                linewidth=2)
    max_val = plot_melt['Magnitude'].max()
    plt.text(2, max_val,
             'HIGH RISK',
             ha='center',
             fontweight='bold',
             color='red')
    plt.text(7, max_val,
             'NORMAL',
             ha='center',
             fontweight='bold',
             color='green')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'customer_comparison.png'))
    plt.close()

if __name__ == '__main__':
    visuals()




