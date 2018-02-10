#include <fstream>
#include <experimental/filesystem>
#include <iostream>

using namespace std::experimental::filesystem;

std::string g_keyword = "front";

void record(const std::string input, const std::string output) {
	std::cout << "reading " << input << "\n\tto " << output << '\n';
	path outputTarget(output);
	if (output.empty()) return;
	if (!exists(outputTarget.parent_path()))
		create_directory(outputTarget.parent_path());

	std::ifstream ifs;
	std::ofstream ofs;
	ifs.open(input);
	ofs.open(output);
	if (ofs.is_open() && ifs.is_open()) {
		int n1, n2, n3, n4, n;
		std::string key, key1, key2;
		char c;
		while (ifs >> key) {
			if (key == g_keyword) {
				ifs >> c >> n1 >> c >> n2 >> c >> key >> n3 >> /*key2 >> n >> key2 >> n >>*/ key2 >> n4;
				ofs << n1 << '\t' << n2 << '\t' << n3 << '\t' << n4 << '\n';
				//ofs << (n1 + n2) * 1. / n3 << '\n';
			}
		}
		ifs.close();
		ofs.close();
	}
}

void scan(const std::string dir) {
	path cates(dir);
	const directory_iterator end{};
	for (directory_iterator it{ cates }; it != end; ++it) {
		if (is_directory(*it)) {
			const auto cateName = it->path().stem().string();
			//std::cout << it->path().string() << ' ' << cateName << '\n';

			for (directory_iterator itr{ it->path() }; itr != end; ++itr) {
				//std::cout << itr->path().string() << '\n';
				if (!is_directory(*itr)) {
					record(itr->path().string(), it->path().parent_path().parent_path().string() + "\\frontstatus\\" + cateName + '\\' + itr->path().stem().string() + ".txt");
				}
			}

		}
	}
}

int main(int argc, char** argv) { 
	if (argc == 1)
		scan("C:\\Mine\\ExpResults\\standaloneCD\\");
	else {
		g_keyword = argv[1];
		scan(argv[2]);
	}

	getchar();
	return 0;
}