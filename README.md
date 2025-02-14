# Marketing BI Template
Coalesce code repository for vertriebswerk BI setup

Use the following template to setup the staging and mart nodes on your repo
Steps to take:

1. Setup fivetran connectors ans sync the raw data with the naming conventions
2. Setup 3 snowflake evironments: Dev, QA and Prod with appropreate user permissions
3. Setup your git repo and coalesce account. Generate the git token and update the snowflake connection on Coalesce
4. Create coalesce Main and User workspaces
5. Clone the template repo into your repo and push changes to main.
6. Refresh the coalesce workspace and view the populated node graphs.

These template nodes are based on multiple marketing reporitng BI sources. They can be modified accordingly.
For the detailed documentation on nodes, view Coalesce site.

