import streamlit as st
import pandas as pd

def load_data():
    # Load CI data
    ci_data = pd.read_csv('data.csv')
    return ci_data

def get_ci_dependencies(ci_name, ci_data):
    ci = ci_data[ci_data['LOGICAL_NAME'] == ci_name].iloc[0]
    if ci['TYPE'] == 'APPLICATION' and ci['SUBTYPE'] == 'DEPLOYMENT':
        dependencies = ci_data[
            (ci_data['R_NETWORK_NAME'] == ci['R_NETWORK_NAME']) &
            (ci_data['TYPE'] != 'APPLICATION')
        ]
        return dependencies[['LOGICAL_NAME', 'TYPE', 'SUBTYPE', 'DESCRIPTION']].to_dict('records')
    return []

def get_ci_downstream_relationships(ci_name, ci_data):
    ci = ci_data[ci_data['LOGICAL_NAME'] == ci_name].iloc[0]
    if ci['TYPE'] != 'APPLICATION' or ci['SUBTYPE'] in ['DEPLOYMENT', 'MASTER']:
        downstream = ci_data[ci_data['LOGICAL_NAME'] == ci_name]
        return downstream[['R_LOGICAL_NAME', 'R_NETWORK_NAME']].to_dict('records')
    return []

def triage_ci(ci_name, ci_data):
    try:
        ci = ci_data[ci_data['LOGICAL_NAME'] == ci_name].iloc[0]
    except IndexError:
        return None, None, None
    
    ci_info = {
        'LOGICAL_NAME': ci['LOGICAL_NAME'],
        'TYPE': ci['TYPE'],
        'SUBTYPE': ci['SUBTYPE'],
        'DESCRIPTION': ci['DESCRIPTION'],
        'NETWORK_NAME': ci['NETWORK_NAME'],
        'LOCATION': ci['LOCATION']
    }
    
    dependencies = get_ci_dependencies(ci_name, ci_data)
    downstream = get_ci_downstream_relationships(ci_name, ci_data)
    
    return ci_info, dependencies, downstream

def main():
    st.title("CI Triage Application")
    
    ci_data = load_data()
    ci_name = st.text_input("Enter the network/application name (LOGICAL_NAME):")
    
    if ci_name:
        ci_info, dependencies, downstream = triage_ci(ci_name, ci_data)
        
        if ci_info:
            st.header("CI Information")
            for key, value in ci_info.items():
                st.write(f"*{key}:* {value}")
            
            st.header("Infrastructure Dependencies")
            if dependencies:
                for dep in dependencies:
                    st.write(f"- *{dep['LOGICAL_NAME']}* ({dep['TYPE']} - {dep['SUBTYPE']}): {dep['DESCRIPTION']}")
            else:
                st.write("No dependencies found.")
            
            st.header("Downstream Relationships")
            if downstream:
                for rel in downstream:
                    st.write(f"- Logical Name: *{rel['R_LOGICAL_NAME']}*, Network Name: {rel['R_NETWORK_NAME']}")
            else:
                st.write("No downstream relationships found.")
        else:
            st.error(f"No CI found with LOGICAL_NAME: {ci_name}")

if __name__ == "__main__":
    main()
