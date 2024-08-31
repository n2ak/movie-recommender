cd front
python -m prisma migrate
npx prisma migrate dev --name init
cd ../back
python filldb.py
cd ..