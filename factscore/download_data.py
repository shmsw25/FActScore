import argparse
import os
import subprocess


def download_file(_id, dest):
    if os.path.exists(dest):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))

if __name__ == '__main__':
    download_file("1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt", "demos.zip")
    download_file("1pUhBqQnrK9ZlgGdpP8LfPaG27yQRSIeT", "original_generation.zip")
    download_file("1mekls6OGOKLmt7gYtHs0WGf5oTamTNat", "enwiki-20230401.db")
    download_file("1TAtyCI75xkqqVlcAFC3pUbsCn8h82vPw", "inst-llama-7B.zip")

    cache_dir = ".cache/factscore"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # move the files to the cache directory
    subprocess.run(["mv demos %s" % cache_dir], shell=True)
    subprocess.run(["mv enwiki-20230401.db %s" % cache_dir], shell=True)