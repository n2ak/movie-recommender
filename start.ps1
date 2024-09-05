cd front
# rm vlms -r -Force
npx prisma migrate dev --name init
python -m prisma migrate
cd ../back
python filldb.py
cd ..