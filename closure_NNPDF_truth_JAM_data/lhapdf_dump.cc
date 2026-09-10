#include <LHAPDF/LHAPDF.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: lhapdf_dump PDFSET MEMBER Q < x_values\n";
    return 2;
  }

  const std::string set_name = argv[1];
  const int member = std::atoi(argv[2]);
  const double q = std::atof(argv[3]);

  std::unique_ptr<LHAPDF::PDF> pdf(LHAPDF::mkPDF(set_name, member));
  std::cout << std::setprecision(17);
  std::cout << "# x g u ubar d dbar s sbar c cbar\n";

  double x = 0.0;
  while (std::cin >> x) {
    std::cout << x
              << " " << pdf->xfxQ(21, x, q)
              << " " << pdf->xfxQ(2, x, q)
              << " " << pdf->xfxQ(-2, x, q)
              << " " << pdf->xfxQ(1, x, q)
              << " " << pdf->xfxQ(-1, x, q)
              << " " << pdf->xfxQ(3, x, q)
              << " " << pdf->xfxQ(-3, x, q)
              << " " << pdf->xfxQ(4, x, q)
              << " " << pdf->xfxQ(-4, x, q)
              << "\n";
  }
  return 0;
}

