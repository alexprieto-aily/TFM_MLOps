import requests
import time
import numpy as np
import json
# import bentoml
import matplotlib.pyplot as plt


url_bentoml = 'http://127.0.0.1:3000/predict'
url_fastAPI = 'http://0.0.0.0:80/predict'


def request_multiple(url, dataframe, n_requests):
    json_payload = dataframe.to_json(orient='records')
    json_data = json.loads(json_payload)  # Load string payload into a list

    if url == url_bentoml:
        json_data = {"loan_data": json_data}

    response_times = []
    for _ in range(n_requests):  # Assuming each row is a request
        
        try:
            start_time = time.time()
            response = requests.post(url, json=json_data, headers={'Content-Type': 'application/json'})
            end_time = time.time()

            response_times.append(end_time - start_time)
            if response.status_code == 200:
                print("Prediction Batch:", response.json())
            else:
                print("Failed to get a response:", response.status_code, response.text)

        except:
            continue
    print_stats(response_times)
    return response_times

def print_stats(response_times):
    print(f"Average Response Time: {sum(response_times) / len(response_times):.4f} seconds")
    print(f"Median Response Time: {np.median(response_times):.4f} seconds")
    print(f"Max Response Time: {max(response_times):.4f} seconds")
    print(f"Min Response Time: {min(response_times):.4f} seconds")





def plot_results(bentoML,fastAPI,add_title):
    # Create x-axis indices for each list
    indices_bentoML = list(range(1, len(bentoML) + 1))
    indices_fastAPI = list(range(1, len(fastAPI) + 1))

    plt.figure(figsize=(10, 5))  # Specify the figure size

    # Plot the list 'bentoML_API'
    plt.plot(indices_bentoML, bentoML,  label='BentoML',  color='blue')

    # Plot the list 'fastAPI'
    plt.plot(indices_fastAPI, fastAPI,  label='FastAPI',  color='red')

    # add median bentoML_API
    plt.axhline(y=np.median(bentoML), color='blue', linestyle='--', label='Median BentoML')

    # add median fastAPI
    plt.axhline(y=np.median(fastAPI), color='red', linestyle='--', label='Median FastAPI')

    plt.title('BentoML vs FastAPI responses time ' + add_title)  # Add a title
    plt.xlabel('Number of Request')  # Add an x-axis label
    plt.ylabel('Time for response')  # Add a y-axis label
    plt.grid(True)  # Show grid
    plt.legend()  # Display the legend with labels
    plt.show()  # Display the plot




def plot_violins(data1, data2, add_title, labels=['Data 1', 'Data 2']):
    fig, ax = plt.subplots()
    ax.violinplot([data1, data2], showmeans=False, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Values')
    plt.title('BentoML vs FastAPI responses time ' + add_title)  # Add a title

    plt.show()


if __name__ == "__main__":

    results_bentoml = request_multiple(url=url_bentoml, width_requests=1, length_requests=1000)
    results_fastAPI = request_multiple(url=url_fastAPI, width_requests=1, length_requests=1000)


    plot_results(bentoML=results_bentoml, fastAPI=results_fastAPI, add_tile=' single')


    results_bentoml = request_multiple(url=url_bentoml, width_requests=10, length_requests=1000)
    results_fastAPI = request_multiple(url=url_fastAPI, width_requests=10, length_requests=1000)

    plot_results(bentoML=results_bentoml, fastAPI=results_fastAPI, add_tile=' multi 10')

    results_bentoml = request_multiple(url=url_bentoml, width_requests=100, length_requests=1000)
    results_fastAPI = request_multiple(url=url_fastAPI, width_requests=100, length_requests=1000)

    plot_results(bentoML=results_bentoml, fastAPI=results_fastAPI, add_tile=' multi 100')




