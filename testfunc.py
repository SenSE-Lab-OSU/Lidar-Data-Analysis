import sys

def main():
    print("Got argument:")
    print(sys.argv[0])
    print(sys.argv[1])
    print(len(sys.argv))

if __name__ == "__main__":
    main()
