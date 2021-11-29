#start service
python remove_background.py

#test service
python test.py
requires: pip install requests




gcloud builds submit --tag gcr.io/remove-background-333611/rb
gcloud run deploy --tag gcr.io/remove-background-333611/rb --platform managed
