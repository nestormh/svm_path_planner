package carrito.gps;

import java.io.File;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;

public class ProcesaImagen {

  static {
    System.loadLibrary("procesaimagen");
  }

  public native int[] _procesaImagenes(double posX1, double posY1, double angulo1,
                             int ancho1, int alto1, int[] imagen1,
                             double posX2, double posY2, double angulo2,
                             int ancho2, int alto2, int[] imagen2);

  public ProcesaImagen() {
  }

  public ImagenId procesaImagenes(ImagenId img1, ImagenId img2) {
    int result[] = this._procesaImagenes(img1.getPosX(), img1.getPosY(), img1.getAngulo(),
                                 img1.getAncho(), img1.getAlto(), img1.getImagen(),
                                 img2.getPosX(), img2.getPosY(), img2.getAngulo(),
                                 img2.getAncho(), img2.getAlto(), img2.getImagen());
    return new ImagenId(0,0,0,(int)Math.min(img1.getAncho(), img2.getAncho()),
                        (int)Math.min(img1.getAlto(), img2.getAlto()), result);
  }

  public static void main(String args[]) {
    ProcesaImagen pi = new ProcesaImagen();
    ImagenId img1 = null, img2 = null;
    try {
      File file = new File("imagen1.img");
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      img1 = (ImagenId)is.readObject();
      is.close();
      file = new File("imagen2.img");
      is = new ObjectInputStream(new FileInputStream(file));
      img2 = (ImagenId)is.readObject();
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }
    if ((img1 != null) && (img2 != null)) {
      ImagenId img3 = pi.procesaImagenes(img1, img2);
      /*frmMuestra frm = new frmMuestra(new CanvasMuestra[] { new CanvasMuestra(img1),
                                      new CanvasMuestra(img2),
                                      new CanvasMuestra(img3) });
      frm.setVisible(true);*/
    }
  }
}
