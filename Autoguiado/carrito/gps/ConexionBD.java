package carrito.gps;

import java.sql.*;
import java.io.InputStream;
import java.io.ObjectInputStream;
import javax.swing.*;

public class ConexionBD {
  private final String driver = "com.mysql.jdbc.Driver";
  String url = "jdbc:mysql://localhost/ruta";
  private final String username = "";
  private final String password = "";

  static final String WRITE_OBJECT_SQL = "insert into ruta(indice, imagen, nombre) values(?, ?, ?);";
  static final String GET_MAX_SQL = "SELECT max(indice) FROM ruta where nombre = ? ;";
  static final String READ_OBJECT_SQL = "SELECT imagen FROM ruta WHERE indice = ? AND nombre = ? ;";

  Connection con = null;

  public void getConnection() {
    try {
      Class.forName(driver);
      con = DriverManager.getConnection(url, username, password);
    } catch(Exception e) {
      System.err.println("Error al recuperar conexion " + e.toString());
      System.exit(0);
    }
  }

  public void stopConnection() {
    try {
      con.close();
    } catch(Exception e) {
      System.err.println("Error al finalizar la conexion " + e.toString());
      System.exit(0);
    }
  }

  public int getMaxIndex(String nombreTabla) {
    if (con == null) {
      System.err.println("No existe conexión con la Base de Datos");
      return -1;
    }
    try {
      PreparedStatement pstmt = con.prepareStatement(GET_MAX_SQL);
      pstmt.setString(1, nombreTabla);
      ResultSet rs = pstmt.executeQuery();
      rs.next();
      int valor = rs.getInt(1);
      rs.close();
      pstmt.close();
      return valor;
    } catch(Exception e) {
      System.err.println("Error al obtener el valor máximo de la Base de Datos " + e.toString());
      return -1;
    }
  }

  public boolean writeObject(Object obj, String nombreTabla) {
    int index;
    if ((index = getMaxIndex(nombreTabla)) == -1)
      return false;
    try {
      con.setAutoCommit(false);
      PreparedStatement pstmt = con.prepareStatement(WRITE_OBJECT_SQL);
      pstmt.setInt(1, index + 1);
      pstmt.setObject(2, obj);
      pstmt.setString(3, nombreTabla);
      pstmt.executeUpdate();

      pstmt.close();

      con.commit();
      con.setAutoCommit(true);
    } catch(Exception e) {
      System.err.println("Error al escribir el objeto, " + e.toString());
      return false;
    }
    return true;
  }

  public Object readObject(int index, String nombreTabla) {
    if (con == null) {
      System.err.println("No existe conexión con la Base de Datos");
      return null;
    }
    try {
      PreparedStatement pstmt = con.prepareStatement(READ_OBJECT_SQL);
      pstmt.setInt(1, index);
      pstmt.setString(2, nombreTabla);

      ResultSet rs = pstmt.executeQuery();
      rs.next();
      Blob campo = rs.getBlob(1);
      InputStream is = campo.getBinaryStream();
      ObjectInputStream ois = new ObjectInputStream(is);
      Object obj = ois.readObject();

      ois.close();
      is.close();
      rs.close();
      pstmt.close();

      return obj;
    } catch(Exception e) {
      System.err.println("Error al leer el objeto, " + e.toString());
      return null;
    }
  }

  public static void main(String args[]) {
    ConexionBD bd = new ConexionBD();
    bd.getConnection();
    int max = bd.getMaxIndex("pruebaBD2");
    CanvasMuestra cm = null;
    for (int i = 1; i < max; i++) {
      ObjetoRuta or = (ObjetoRuta)bd.readObject(i, "pruebaBD2");
      System.out.println(i + "-->" + max);
      System.out.println("X: " + or.getX());
      System.out.println("Y: " + or.getY());
      System.out.println("Z: " + or.getZ());
      System.out.println("Angulo: " + or.getAngulo());
      System.out.println("Vel: " + or.getSpeed());
      System.out.println("(" + (int)or.getDim1().getWidth() + ", " + (int)or.getDim1().getHeight() + ")");
      System.out.println("(" + (int)or.getDim2().getWidth() + ", " + (int)or.getDim2().getHeight() + ")");
      if (or.getImg2() == null)
        System.out.println("Imagen nula");

      ImagenId ii = new ImagenId(or.getX(), or.getY(), or.getZ(),
                                 (int)or.getDim1().getWidth(), (int)or.getDim1().getHeight(), or.getImg1());
      if (cm == null)
        cm = new CanvasMuestra(ii);
      cm.setImagen(ii);
      cm.repaint();
      try {
        Thread.sleep(10);
      } catch(Exception e) {}
    }
    bd.stopConnection();
  }
}
