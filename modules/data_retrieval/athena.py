# %% imports
import os
import boto3
import pandas as pd


# %% main functions
def run_athena_query(
    query_string: str,
    output_location: str,
    region: str = 'eu-central-1'
):
    athena_client = boto3.client('athena', region_name=region)

    response = athena_client.start_query_execution(
        QueryString=query_string,
        ResultConfiguration={
            'OutputLocation': output_location,
        }
    )

    query_execution_id = response['QueryExecutionId']
    print(f"Query Execution ID: {query_execution_id}")

    # Wait for the query to complete
    while athena_client.get_query_execution(
        QueryExecutionId=query_execution_id
    )['QueryExecution']['Status']['State'] not in [
        'SUCCEEDED', 'FAILED', 'CANCELLED'
    ]:
        continue

    return query_execution_id


def fetch_results(
    query_execution_id: int,
    output_location: str,
    region: str = 'eu-central-1'
) -> pd.DataFrame:
    athena_client = boto3.client('athena', region_name=region)

    # Wait for the query to complete
    while athena_client.get_query_execution(
        QueryExecutionId=query_execution_id
    )['QueryExecution']['Status']['State'] not in [
        'SUCCEEDED', 'FAILED', 'CANCELLED'
    ]:
        continue

    csv_path = os.path.join(output_location, query_execution_id + '.csv')

    for k in range(200):
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            continue
