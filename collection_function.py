import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########## Data Transformation ##########

def read_file(csv_file):
  """ Reading CSV File """
  data = pd.read_csv(csv_file)
  return data

def simple_pivot(data, ind, val, agg):
   """
   Pivoting data and transform into dataframe
   """
   pivot = pd.pivot_table(data, index=ind, values=val, aggfunc=agg).reset_index()
   pivot = pivot.sort_values(by=ind, ascending=False)

   return pivot

def calculate_percent_of_total(data, subset, val):
    """
    This function is to calculate the percent of total in subset
    """
    data['total_subset'] = data.groupby(subset)[val].transform('sum')
    data['percent_of_total'] = data[val]/data['total_subset']*100
    return data

def calculate_cumsum(data, subset, val):
    """
    This function is to calculate the cumulative percent of total in subset
    """
    data['cumsum'] = data.groupby(subset)[val].cumsum()
    return data

def check_null_values(data, cols):
    """
    This function aims to calculate the null values and its percentage 
    """
    #particular columns
    check_null = data[cols]

    #show null values and its percentage
    null_val = pd.DataFrame(check_null.dtypes).T.rename(index = {0:'Columns Type'})
    null_val = pd.concat([null_val, pd.DataFrame(check_null.isnull().sum()).T.rename(index = {0:'Amount of Null Values'})])
    null_val = pd.concat([null_val, pd.DataFrame(round(check_null.isnull().sum()/check_null.shape[0]*100,2)).T.rename(index = {0:'Percentage of Null Values'})])
    return null_val.T

def label_missing(row):
    """
    Check if any value in the row is NaN
    """
    if row.isnull().any():
        return 'Missing'
    else:
        return 'Complete'


def taking_outlier_data(data, col):
    """
    Taking outlier data

    data = dataset
    col = desired column
    """
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    return data

########## Data Visualization ##########

def distribution(dataset, val, xlabel, ylabel, title):
  """Visualizign the Distribution of Numerical data"""
  # Calculate mean and median
  mean_value = dataset[val].mean()
  median_value = dataset[val].median()
  min_value = dataset[val].min()
  max_value = dataset[val].max()

  # Plotting the distribution
  plt.figure(figsize=(6, 4))
  sns.histplot(dataset[val], kde=True, bins=10, color='blue')
  plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
  plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')
  plt.axvline(min_value, color='blue', linestyle='-', label=f'min: {min_value:.2f}')
  plt.axvline(max_value, color='blue', linestyle='-', label=f',max: {max_value:.2f}')

  # Adding labels and title
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend()

  # Show the plot
  return plt.show()

def bar_chart(data, x_vals, y_vals, title, xlabel, ylabel, ymax):
    """
    Creating bar char
    """

    # Extract x-axis and y-axis values
    x = data[x_vals]  # Categories (e.g., labels)
    y = data[y_vals]  # Values (e.g., counts)

    # Create the bar plot
    plt.figure(figsize=(12, 5))
    ax = plt.bar(x, y, color='lightcoral', edgecolor='black')

    # Calculate total for percentage calculation
    total_count = y.sum()

    # Add data labels
    for idx, value in enumerate(y):
        percent = (value / total_count) * 100
        plt.text(idx, value, f'{value}\n({percent:.2f}%)', ha='center', va='bottom', fontsize=10)
        # plt.text(value, f'{percent:.2f}%', ha='center', va='bottom', fontsize=10)

    # Add title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=0)
    # plt.tight_layout()

    # Set specific y-axis limits
    plt.ylim(0, y.max() + ymax)

    # Show the plot
    return plt.show()

def line_chart(data, x_vals, y_vals, title, xlabel, ylabel):
    """
    Creating line chart
    """
    plt.figure(figsize=(8, 4))
    plt.plot(data[x_vals], data[y_vals], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    # Show the plot
    return plt.show()

def horizontal_bar_chart(data, x, y, z, xlabel, ylabel, title):
    """
    This horizontal bar chart aims to inform the absolute and percent of total in each item
    """
    # Plotting a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = plt.barh(data[y], data[x], color='skyblue', label='Number of Agent')

    # Adding annotations
    for bar, percentage in zip(bars, data[z]):
        width = bar.get_width()
        ax.annotate(f'{int(width)} ({percentage:.2f}%)', 
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # offset text slightly to the right of the bar
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9)

    # Add labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, data[x].max()+data[x].max()*0.2)

    # Show plot
    return plt.show()

def horizontal_bar_chart_v2(data, x, y, z, xlabel, ylabel, title, xmin, xmax):
    # Set up figure size
    plt.figure(figsize=(10, 6))

    # Create horizontal bar chart
    sns.barplot(x=x, y=y, data=data, palette="Blues")

    # Add data labels with absolute values and percent_of_total
    for index, row in data.iterrows():
        plt.text(row[x], index, 
                f'{abs(row[x]):,} ({abs(row[z])}%)', 
                va='center',
                fontsize=10, fontweight="bold", color="black")

    # Title and labels
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(data[x].min() + data[x].min()*xmin, 
            data[x].max() + data[x].max()*xmax)

    # Show grid for better readability
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    # Display the chart
    plt.show()


def pie_chart(data, val, cat, title):
    """
    Function to visualize the composition of data with a blue gradient color.
    """
    # Function to display both absolute numbers and percentages
    def func(pct, all_values):
        absolute = int(pct / 100. * sum(all_values))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    # Generate gradient colors from the "Blues" colormap
    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(data[val]) + 0.4) for i in range(len(data[val]))]  # Adjust brightness

    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        data[val], labels=data[cat], colors=colors, 
        autopct=lambda pct: func(pct, data[val]), startangle=90, counterclock=False
    )

    # Style autopct text
    for autotext in autotexts:
        autotext.set_color("white")  # Make text more readable on darker shades
        autotext.set_fontsize(10)

    # Ensure equal aspect ratio
    plt.axis('equal')
    plt.title(title)

    # Show the chart
    plt.show()

def distribution_multiple_values(data, label, title):
    """
    This function is used if you want to show the distribution with multiple values
    """
    # List of data point
    data_ = data

    # Colors for each category (expand as needed)
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # custom labels
    labels = label

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Loop through each category and plot
    for i, category in enumerate(data_):
        label = labels[i] if i < len(labels) else f'Category {i + 1}'

        sns.histplot(category, bins=30, color=colors[i % len(colors)], alpha=0.4, kde=True, label=label)
        plt.axvline(np.mean(category), color=colors[i % len(colors)], linestyle='dashed', linewidth=2, label=f'Mean ({label})')
        plt.axvline(np.median(category), color=colors[i % len(colors)], linestyle='solid', linewidth=2, label=f'Median ({label})')

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1))

    # Set plot title and labels
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

def heatmap_corr(data, title, method):
    """
    Function to calculate and visualize the correlation matrix of a DataFrame.
    
    Parameters:
    - data: pd.DataFrame, the input data
    - method: str, the method of correlation ('pearson', 'kendall', 'spearman')
    
    Returns:
    - corr_matrix: pd.DataFrame, the correlation matrix
    """

    # Calculate the correlation matrix
    correlation_matrix = data.corr(method=method)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(title)
    return plt.show()

def gradient_background(data, subset, color, formatting):
    """
    Provide gradient background to highlight the information

    example: data.style.background_gradient(subset=['percent_of_total'], cmap='Reds').format({'percent_of_total': "{:.2f}"})
    """
    gradient = data.style.background_gradient(subset=subset, cmap=color).format(formatting)

    return gradient

def stacked_bar_chart(data, ind, col, val, title, xlabel, ylabel, legend):
    """
    This function is used to visualize the stacked bar chart
    """

    # Pivot the data for a stacked bar chart
    pivot_cumsum = data.pivot_table(index=ind, columns=col, values=val, aggfunc='count')

    pivot_cumsum = pivot_cumsum.sort_values(by=pivot_cumsum.index[0], axis=1, ascending=False)
    pivot_cumsum = pivot_cumsum.fillna(0)

    # Calculate row-wise total for percentage calculation
    row_totals = pivot_cumsum.sum(axis=1)

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom_value = [0] * len(pivot_cumsum)  # Initialize bottom values for stacking

    # Dynamically generate colors for all product types
    colors = sns.color_palette("muted", n_colors=len(pivot_cumsum.columns))

    # Use the color for each product type
    for idx, product in enumerate(pivot_cumsum.columns):
        values = pivot_cumsum[product].values
        sns.barplot(
            x=pivot_cumsum.index,
            y=values,
            label=product,
            ax=ax,
            color=colors[idx],
            bottom=bottom_value
        )
        
        # Add data labels for each bar segment
        for i, value in enumerate(values):
            if value > 0:
                percent = (value / row_totals.iloc[i]) * 100
                label = f"{int(value)} ({percent:.1f}%)"
                ax.text(
                    i,
                    bottom_value[i] + value / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if percent > 10 else "black"  # Optional: contrast for readability
                )

        # Update bottom values for stacking
        bottom_value = [bottom_value[j] + values[j] for j in range(len(values))]

    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title=legend, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Show the plot
    return plt.show()
