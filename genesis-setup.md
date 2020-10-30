# Genesis compute QbE-STD setup

## Set up

- Set time zone to Pacific (or whatever) for logging

	```
	sudo timedatectl set-timezone America/Los_Angeles
	```

- Install packages

	```
	sudo apt-get update
	sudo apt-get install -y unzip
	```

- Install task and copy to `/usr/bin`

	``` 
	curl -sL https://taskfile.dev/install.sh | sh
	sudo cp bin/task /usr/bin
	```

- Install rclone and configure

	```
	curl https://rclone.org/install.sh | sudo bash
	```
	
### Fetch data
	
- Get repo and data

	```
	git clone https://github.com/fauxneticien/bnf_cnn_qbe-std.git
	cd bnf_cnn_qbe-std
	rclone copy --progress gdrive:cnn/data/sws2013train.zip data
	unzip data/sws2013train.zip -d data/sws2013
	```

## Upload artifacts

- Sync artifacts to Google Drive

	```
	rclone sync tmp gdrive:/cnn/genesis-instance/tmp
	```
	