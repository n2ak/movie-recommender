#!/bin/bash
set -e

if [ -n "$DB_USERS" ] && [ -n "$DB_PASSWORDS" ] && [ -n "$DB_DBS" ]; then
  USERS=($DB_USERS)
  PASSWORDS=($DB_PASSWORDS)
  DBS=($DB_DBS)

  for i in "${!USERS[@]}"; do
    user="${USERS[$i]}"
    pass="${PASSWORDS[$i]}"
    db="${DBS[$i]}"

    echo "Creating user '$user' with database '$db'"

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
      CREATE USER $user WITH PASSWORD '$pass';
      CREATE DATABASE $db OWNER $user;
      GRANT ALL PRIVILEGES ON DATABASE $db TO $user;
EOSQL
  done
else
  echo "DB_USERS, DB_PASSWORDS, or DB_DBS not set — skipping extra user creation."
fi
