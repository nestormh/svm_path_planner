package repgrafica;

import sibtra.gps.Trayectoria;
//import sibtra.lms.BarridoAngular;
//import sibtra.lms.ManejaLMS;
import sibtra.lms.PanelMuestraBarrido;
import sibtra.lms.ManejaLMS111;
import sibtra.log.LoggerFactory;
import sibtra.log.VentanaLoggers;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
//import java.awt.geom.Line2D.Double;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sun.reflect.ReflectionFactory.GetReflectionFactoryAction;

import com.bruceeckel.swing.Console;

import Jama.Matrix;
import boids.*;

class Dibujante2 extends JPanel{
	double largoCoche = 1.7;
	double anchoCoche = 1;
	Matrix posCoche = new Matrix(2,1);
	double posCocheX = 0;
	double posCocheY = 0;
	double posCocheAEstrellaX = 0;
	double posCocheAEstrellaY = 0;
	Matrix posCocheSolitario = new Matrix(2,1);
	double posCocheSolitarioX = 0;
	double posCocheSolitarioY = 0;
	Matrix vectorDirectorCoche = new Matrix(2,1);
	double yawCoche;
	double yawCocheAEstrella;
	double yawCocheSolitario;
	Vector<Boid> bandadaPintar;// = new Vector<Boid>();
	Vector<Obstaculo> obstaculosPintar;// = new Vector<Obstaculo>();
	Vector<Matrix> rutaDinamica = null;
	Vector<Matrix> rutaAEstrella = null;	
	double [][] prediccion;
	int horPrediccion;
	
	public Dibujante2(){
		bandadaPintar = new Vector<Boid>();
		obstaculosPintar = new Vector<Obstaculo>();
	}
	public Matrix getVectorDirectorCoche() {
		return vectorDirectorCoche;
	}

	public void setVectorDirectorCoche(Matrix vectorDirectorCoche) {
		this.vectorDirectorCoche = vectorDirectorCoche;
	}
	
	public double getYawCoche() {
		return yawCoche;
	}

	public void setYawCoche(double yawCoche) {
		this.yawCoche = yawCoche;
	}
	
	public double getYawCocheAEstrella() {
		return yawCocheAEstrella;
	}
	public void setYawCocheAEstrella(double yawCocheAEstrella) {
		this.yawCocheAEstrella = yawCocheAEstrella;
	}

	public double getYawCocheSolitario() {
		return yawCocheSolitario;
	}

	public void setYawCocheSolitario(double yawCocheSolitario) {
		this.yawCocheSolitario = yawCocheSolitario;
	}

	public Matrix getPosCoche() {
		return posCoche;
	}

	public void setPosCoche(Matrix posIni) {
		this.posCoche = posIni;
		this.posCocheX = posIni.get(0, 0);
		this.posCocheY = posIni.get(1, 0);
	}
	
	public void setPosCoche(double x,double y) {
		this.posCocheX = x;
		this.posCocheY = y;
		this.posCoche.set(0, 0, x);
		this.posCoche.set(1, 0, y);
	}
	
	public Matrix getPosCocheSolitario() {
		return posCocheSolitario;
	}

	public void setPosCocheSolitario(Matrix posCocheSolitario) {
		this.posCocheSolitario = posCocheSolitario;
		this.posCocheSolitarioX = posCocheSolitario.get(0, 0);
		this.posCocheSolitarioY = posCocheSolitario.get(1, 0);
	}
	
	public void setPosCocheSolitario(double x,double y) {		
		this.posCocheSolitarioX = x;
		this.posCocheSolitarioY = y;
		this.posCocheSolitario.set(0, 0, x);
		this.posCocheSolitario.set(1, 0, y);
	}
	
	public void setPosCocheAEstrella(double x,double y) {		
		this.posCocheAEstrellaX = x;
		this.posCocheAEstrellaY = y;
	}
	
	public void introducirBoid(Boid b){
		bandadaPintar.add(b);		
	}
	
	public void introducirBandada(Vector<Boid> banda){
		bandadaPintar = banda;
//		bandadaPintar.clear();
//		for (int i =0;i<banda.size();i++){
//			bandadaPintar.add(banda.elementAt(i));
//		}
	}
	
	public void introducirObstaculo(Obstaculo b){
		obstaculosPintar.add(b);		
	}
	
	public void introducirObstaculos(Vector<Obstaculo> banda){
		for (int i =0;i<banda.size();i++){
			obstaculosPintar.add(banda.elementAt(i));
		}
	}
	
	public void eliminarObstaculos(){
		obstaculosPintar.clear();
	}
	
	public Vector<Boid> getBandadaPintar(){
		return bandadaPintar;
	}
	public Vector<Obstaculo> getObstaculoPintar(){
		return obstaculosPintar;
	}
	
	public void setRutaDinamica(Vector<Matrix> ruta){
		this.rutaDinamica = ruta;
	}
	
	public Vector<Matrix> getRutaAEstrella() {
		return rutaAEstrella;
	}
	public void setRutaAEstrella(Vector<Matrix> rutaAEstrella) {
		this.rutaAEstrella = rutaAEstrella;
	}
	
	public void setPrediccion(double[][] prediccionPosPorFilas) {
		System.out.println("predicción pasada al dibujante");
		this.prediccion = prediccionPosPorFilas;		
	}
	
	public int getHorPrediccion() {
		return horPrediccion;
	}
	public void setHorPrediccion(int horPrediccion) {
		this.horPrediccion = horPrediccion;
	}
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	public Point2D esqSuperiorIzquierda;
	
		/**
	 * Coordenadas de la esquina inferior derecha.
	 * En unidades mundo real.
	 */
	public Point2D esqInferiorDerecha;
	
	public Point2D getEsqSuperiorIzquierda() {
		return esqSuperiorIzquierda;
	}

	public void setEsqSuperiorIzquierda(Point2D esqSuperiorIzquierda) {
		this.esqSuperiorIzquierda = esqSuperiorIzquierda;
	}
	
	public void setEsqSuperiorIzquierda(double x, double y) {
		this.esqSuperiorIzquierda = new Point2D.Double(x,y);
	}

	public Point2D getEsqInferiorDerecha() {
		return esqInferiorDerecha;
	}

	public void setEsqInferiorDerecha(Point2D esqInferiorDerecha) {
		this.esqInferiorDerecha = esqInferiorDerecha;
	}

	public void setEsqInferiorDerecha(double x, double y) {
		this.esqInferiorDerecha = new Point2D.Double(x,y);
	}
	
	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 * @param x coordenada X del punto
	 * @param y coordenada Y del punto
	 * @return punto en pantalla
	 */
	protected Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(x*this.getWidth()/esqInferiorDerecha.getX(),
				this.getHeight()-(y*(this.getHeight()/esqSuperiorIzquierda.getY())));
	}
	/**
	 * Convierte punto de la pantalla a coordenadas reales
	 * @param x medida en pÃ­xeles de x
	 * @param y medida en pÃ­xeles de y
	 * @return
	 */
	protected Point2D.Double pixel2Point(int x, int y){
		return new Point2D.Double(x*esqInferiorDerecha.getX()/this.getWidth(),
				esqSuperiorIzquierda.getY()-(y*(esqSuperiorIzquierda.getY()/this.getHeight())));
	}
	
//	private int incrNuevosBoids = 3;
//	private int contNuevosBoids = incrNuevosBoids ;
	private int contIteraciones = 0;

	
	public void paintComponent(Graphics g2) {
		Graphics2D g3 = (Graphics2D) g2;		
		super.paintComponent(g3);
		Matrix centroMasa = new Matrix(2,1);
		
		//------------------------- Pinto los Boids----------------------------
		
		if (bandadaPintar.size() > 0){		
			contIteraciones++;
			for (int i=0;i<bandadaPintar.size();i++){
		
				g3.setColor(Color.blue);

				Point2D pixel = point2Pixel(bandadaPintar.elementAt(i).getPosicion().get(0,0),
						bandadaPintar.elementAt(i).getPosicion().get(1,0));
				g3.drawOval((int)pixel.getX()-2,(int)pixel.getY()-2,4,4);
//				g3.drawOval((int)bandadaPintar.elementAt(i).getPosicion().get(0,0)-2,
//						(int)bandadaPintar.elementAt(i).getPosicion().get(1,0)-2,4,4);
				
//				g3.draw(bandadaPintar.elementAt(i).getLineaDireccion());
//				GeneralPath ruta = new GeneralPath();
//				Point2D pixelCoche = point2Pixel(posCoche.get(0, 0),posCoche.get(1, 0));
//				ruta.moveTo(pixelCoche.getX(),pixelCoche.getY());
//				for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
//					pixel = point2Pixel(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0, 0),
//							bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1, 0));
//				ruta.lineTo(pixel.getX(),pixel.getY());
//				}
//				g3.draw(ruta);
//				centroMasa = centroMasa.plus(bandadaPintar.elementAt(i).getPosicion());
			}
		}
		
		/*------------------------Pinto el coche----------------------------------*/

		g3.setColor(Color.magenta);

		Point2D pxDD= point2Pixel(posCoche.get(0,0)+anchoCoche/2*Math.sin(yawCoche),
				posCoche.get(1,0)-anchoCoche/2*Math.cos(yawCoche));
		Point2D pxDI= point2Pixel(posCoche.get(0,0)-anchoCoche/2*Math.sin(yawCoche),
				posCoche.get(1,0)+anchoCoche/2*Math.cos(yawCoche));
		Point2D pxPD=point2Pixel(posCoche.get(0,0)+anchoCoche/2*Math.sin(yawCoche)
				-largoCoche*Math.cos(yawCoche),
				posCoche.get(1,0)-anchoCoche/2*Math.cos(yawCoche)
				-largoCoche*Math.sin(yawCoche));
		Point2D pxPI= point2Pixel(posCoche.get(0,0)-anchoCoche/2*Math.sin(yawCoche)
				-largoCoche*Math.cos(yawCoche),
				posCoche.get(1,0)+anchoCoche/2*Math.cos(yawCoche)
				-largoCoche*Math.sin(yawCoche));
		GeneralPath coche=new GeneralPath();
		coche.moveTo((float)pxDD.getX(),(float)pxDD.getY());
		coche.lineTo((float)pxPD.getX(),(float)pxPD.getY());
		coche.lineTo((float)pxPI.getX(),(float)pxPI.getY());
		coche.lineTo((float)pxDI.getX(),(float)pxDI.getY());
		coche.closePath();		
		g3.draw(coche);
		g3.fill(coche);
		
		/*------------------------Pinto el coche A estrella----------------------------------*/
		
		g3.setColor(Color.blue);

		Point2D pxDD2= point2Pixel(posCocheAEstrellaX+anchoCoche/2*Math.sin(yawCocheAEstrella),
				posCocheAEstrellaY-anchoCoche/2*Math.cos(yawCocheAEstrella));
		Point2D pxDI2= point2Pixel(posCocheAEstrellaX-anchoCoche/2*Math.sin(yawCocheAEstrella),
				posCocheAEstrellaY+anchoCoche/2*Math.cos(yawCocheAEstrella));
		Point2D pxPD2=point2Pixel(posCocheAEstrellaX+anchoCoche/2*Math.sin(yawCocheAEstrella)
				-largoCoche*Math.cos(yawCocheAEstrella),
				posCocheAEstrellaY-anchoCoche/2*Math.cos(yawCocheAEstrella)
				-largoCoche*Math.sin(yawCocheAEstrella));
		Point2D pxPI2= point2Pixel(posCocheAEstrellaX-anchoCoche/2*Math.sin(yawCocheAEstrella)
				-largoCoche*Math.cos(yawCocheAEstrella),
				posCocheAEstrellaY+anchoCoche/2*Math.cos(yawCocheAEstrella)
				-largoCoche*Math.sin(yawCocheAEstrella));
		GeneralPath cocheAEstrella=new GeneralPath();
		cocheAEstrella.moveTo((float)pxDD2.getX(),(float)pxDD2.getY());
		cocheAEstrella.lineTo((float)pxPD2.getX(),(float)pxPD2.getY());
		cocheAEstrella.lineTo((float)pxPI2.getX(),(float)pxPI2.getY());
		cocheAEstrella.lineTo((float)pxDI2.getX(),(float)pxDI2.getY());
		cocheAEstrella.closePath();		
		g3.draw(cocheAEstrella);
		g3.fill(cocheAEstrella);
		
		/*----------------------Pinto el coche solitario---------------------------------*/
		g3.setColor(Color.green);

		Point2D pxDD3= point2Pixel(posCocheSolitario.get(0,0)+anchoCoche/2*Math.sin(yawCocheSolitario),
				posCocheSolitario.get(1,0)-anchoCoche/2*Math.cos(yawCocheSolitario));
		Point2D pxDI3= point2Pixel(posCocheSolitario.get(0,0)-anchoCoche/2*Math.sin(yawCocheSolitario),
				posCocheSolitario.get(1,0)+anchoCoche/2*Math.cos(yawCocheSolitario));
		Point2D pxPD3=point2Pixel(posCocheSolitario.get(0,0)+anchoCoche/2*Math.sin(yawCocheSolitario)
				-largoCoche*Math.cos(yawCocheSolitario),
				posCocheSolitario.get(1,0)-anchoCoche/2*Math.cos(yawCocheSolitario)
				-largoCoche*Math.sin(yawCocheSolitario));
		Point2D pxPI3= point2Pixel(posCocheSolitario.get(0,0)-anchoCoche/2*Math.sin(yawCocheSolitario)
				-largoCoche*Math.cos(yawCocheSolitario),
				posCocheSolitario.get(1,0)+anchoCoche/2*Math.cos(yawCocheSolitario)
				-largoCoche*Math.sin(yawCocheSolitario));
		GeneralPath cocheSolitario=new GeneralPath();
		cocheSolitario.moveTo((float)pxDD3.getX(),(float)pxDD3.getY());
		cocheSolitario.lineTo((float)pxPD3.getX(),(float)pxPD3.getY());
		cocheSolitario.lineTo((float)pxPI3.getX(),(float)pxPI3.getY());
		cocheSolitario.lineTo((float)pxDI3.getX(),(float)pxDI3.getY());
		cocheSolitario.closePath();
		g3.draw(cocheSolitario);
		g3.fill(cocheSolitario);
		Point2D vertice1 = point2Pixel(posCocheSolitario.get(0,0),posCocheSolitario.get(1,0));
		Point2D vertice2 = point2Pixel(vectorDirectorCoche.get(0,0)+posCocheSolitario.get(0,0),
				vectorDirectorCoche.get(1,0)+posCocheSolitario.get(1,0));
		GeneralPath vecCocheSolitario = new GeneralPath();
		vecCocheSolitario.moveTo((float)vertice1.getX(),(float)vertice1.getY());
		vecCocheSolitario.lineTo((float)vertice2.getX(),(float)vertice2.getY());
		vecCocheSolitario.closePath();
		g3.setColor(Color.BLACK);
		g3.draw(vecCocheSolitario);
		g3.fill(vecCocheSolitario);
//		//---------------------- Pinto el centro de masa--------------------------------
		
//		centroMasa.timesEquals((double)1/(double)bandadaPintar.size());		
//		Point2D centroMasaPixel = point2Pixel(centroMasa.get(0, 0),
//				centroMasa.get(1, 0));
//		g2.setColor(Color.cyan);
//		g2.drawOval((int)centroMasaPixel.getX()-2,(int)centroMasaPixel.getY()-2,4,4);
		
		//------------------------ Pinto los obstaculos---------------------------------
		
		for (int i=0;i<obstaculosPintar.size();i++){
			if(!obstaculosPintar.elementAt(i).isVisible()){
				g3.setColor(Color.gray);
			}else{
				g3.setColor(Color.red);
			}
			
			Point2D obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-
					obstaculosPintar.elementAt(i).getLado()/2,
					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+
					obstaculosPintar.elementAt(i).getLado()/2);
			double lado = obstaculosPintar.elementAt(i).getLado()*this.getWidth()/esqInferiorDerecha.getX();
			Rectangle2D cuadrado = new Rectangle2D.Double(obst.getX(),
					obst.getY(),lado,lado);
			g3.draw(cuadrado);
			g3.fill(cuadrado);
			obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-Boid.getRadioObstaculo(),
					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+Boid.getRadioObstaculo());
			double radio = Boid.getRadioObstaculo()*this.getWidth()/esqInferiorDerecha.getX();
			g3.drawOval((int)obst.getX(),(int)obst.getY(),(int)radio*2,(int)radio*2);
//			obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-Boid.getRadioObstaculoLejos(),
//					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+Boid.getRadioObstaculoLejos());
//			g3.setColor(Color.black);
//			radio = Boid.getRadioObstaculoLejos()*this.getWidth()/esqInferiorDerecha.getX();
//			g3.drawOval((int)obst.getX(),(int)obst.getY(),(int)radio*2,(int)radio*2);
			
		}
		//---------------------------- Pinto el objetivo-------------------------------------
		g3.setColor(Color.magenta);
		Point2D objetivo = point2Pixel(Boid.getObjetivo().get(0,0),Boid.getObjetivo().get(1,0));
		g3.drawOval((int)objetivo.getX(),(int)objetivo.getY(),5,5);
		
		//----------------------- Pinto la ruta dinÃ¡mica---------------------------------
		
		if (rutaDinamica != null){
			if (rutaDinamica.size() > 0){
				
				GeneralPath rutaDinamic = new GeneralPath();
				Point2D ptoRuta = point2Pixel(rutaDinamica.elementAt(0).get(0,0),
						rutaDinamica.elementAt(0).get(1,0));
				rutaDinamic.moveTo(ptoRuta.getX(),ptoRuta.getY());
				for(int k=1;k<rutaDinamica.size();k++){
					ptoRuta = point2Pixel(rutaDinamica.elementAt(k).get(0,0),
							rutaDinamica.elementAt(k).get(1,0));
					rutaDinamic.lineTo(ptoRuta.getX(),ptoRuta.getY());
				}
				g3.draw(rutaDinamic);
			}
		}
		
		//----------------------- Pinto la ruta a estrella calculada con el grid---------------------------------
		g3.setColor(Color.blue);
		
		if (rutaAEstrella != null){
			if (rutaAEstrella.size() > 0){
				
				GeneralPath rutaDinamic = new GeneralPath();
				Point2D ptoRuta = point2Pixel(rutaAEstrella.elementAt(0).get(0,0),
						rutaAEstrella.elementAt(0).get(1,0));
				rutaDinamic.moveTo(ptoRuta.getX(),ptoRuta.getY());
				for(int k=1;k<rutaAEstrella.size();k++){
					ptoRuta = point2Pixel(rutaAEstrella.elementAt(k).get(0,0),
							rutaAEstrella.elementAt(k).get(1,0));
					rutaDinamic.lineTo(ptoRuta.getX(),ptoRuta.getY());
				}
				g3.draw(rutaDinamic);
			}
		}
		
		//------------------------Pinto la predicción del movimiento del vehículo-------------------
		g3.setColor(Color.black);
		if (prediccion != null){
//			if (prediccion. >= 2){		
				System.out.println("pintando la predicción y el tamaño de la predicción es " + prediccion.length);
				GeneralPath predicc = new GeneralPath();
				Point2D ptoRuta = point2Pixel(prediccion[0][0],prediccion[1][0]);
				predicc.moveTo(ptoRuta.getX(),ptoRuta.getY());
				for(int k=1;k<horPrediccion;k++){
					ptoRuta = point2Pixel(prediccion[0][k],prediccion[1][k]);
					predicc.lineTo(ptoRuta.getX(),ptoRuta.getY());
				}
				g3.draw(predicc);
//			}				
		}
	}		
}


public class MuestraBoids extends JApplet implements ChangeListener,ActionListener,MouseListener{
	
	/**
	 * tiempo que pasa entre pintado y pintado de la escena, siempre y cuando el tiempo de cómputo sea 
	 * más pequeño que este valor. En milisegundos	
	 */
	long velReprod = 2;	
	
	public long getVelReprod() {
		return velReprod;
	}

	public void setVelReprod(long velReprod) {
		this.velReprod = velReprod;
	}
	
	public Vector<Matrix> rutaDinamica;
	public Vector<Matrix> getRutaDinamica() {
		return rutaDinamica;
	}

	public void setRutaDinamica(Vector<Matrix> rutaDinamica) {
		this.rutaDinamica = rutaDinamica;
	}
	
	boolean batch = false;
	private boolean objetivoEncontrado;
	private boolean play = false;
	private boolean playObs = false;
	private boolean colocandoObs = false;
	private boolean colocandoBan = false;
	private boolean pintarEscena = false;
	VentanaLoggers ventLogs = new VentanaLoggers();
	Simulador sim;// = new Simulador();
	JFileChooser selectorArchivo = new JFileChooser(new File("./Escenarios"));
	JMenuBar barraMenu = new JMenuBar(); 
	JMenu menuArchivo = new JMenu("File");
	JMenu menuBandada = new JMenu("Flock");
	Dibujante2 pintor;// = new Dibujante2();
//	JLabel etiquetaPesoLider = new JLabel("Liderazgo");
//	SpinnerNumberModel spPesoLider = new SpinnerNumberModel(Boid.getPesoLider(),0,100,0.1);
//	JSpinner spinnerPesoLider = new JSpinner(spPesoLider);
	JLabel etiquetaVelReprod = new JLabel("Vel reprod");
	SpinnerNumberModel spVelReprod = new SpinnerNumberModel(velReprod,0,10000,1);
	JSpinner spinnerVelReprod = new JSpinner(spVelReprod);
	JLabel etiquetaCohesion = new JLabel("Cohesion");
	SpinnerNumberModel spCohesion = new SpinnerNumberModel(Boid.getRadioCohesion(),0,100,0.1);
	JSpinner spinnerCohesion = new JSpinner(spCohesion);
	JLabel etiquetaSeparacion = new JLabel("Separation");
	SpinnerNumberModel spSeparacion = new SpinnerNumberModel(Boid.getRadioSeparacion(),0,1000,0.1);
	JSpinner spinnerSeparacion = new JSpinner(spSeparacion);
	JLabel etiquetaAlineacion = new JLabel("Alignement");
	SpinnerNumberModel spAlineacion = new SpinnerNumberModel(Boid.getRadioAlineacion(),0,100,0.1);
	JSpinner spinnerAlineacion = new JSpinner(spAlineacion);
//	JLabel etiquetaObjetivo = new JLabel("Vel Obj");
//	SpinnerNumberModel spObjetivo = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,100,0.1);
//	JSpinner spinnerObjetivo = new JSpinner(spObjetivo);
//	JLabel etiquetaEvitaObs = new JLabel("Evitar Obst");
//	SpinnerNumberModel spEvitaObs = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
//	JSpinner spinnerEvitaObs = new JSpinner(spEvitaObs);
//	JLabel etiquetaObsCerca = new JLabel("Obst cerca");
//	SpinnerNumberModel spEvitaObsCerca = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
//	JSpinner spinnerEvitaObsCerca = new JSpinner(spEvitaObsCerca);
//	JLabel etiquetaCompLateral = new JLabel("Comp lateral");
//	SpinnerNumberModel spCompLateral = new SpinnerNumberModel(Boid.getPesoCompensacionLateral(),0,1000,0.1);
//	JSpinner spinnerCompLateral = new JSpinner(spCompLateral);
	JLabel etiquetaRadObs = new JLabel("Obstacle Radius");
	SpinnerNumberModel spRadioObs = new SpinnerNumberModel(Boid.getRadioObstaculo(),0,1000,0.1);
	JSpinner spinnerRadioObs = new JSpinner(spRadioObs);
//	JLabel etiquetaRadObsLejos = new JLabel("Radio lejos");
//	SpinnerNumberModel spRadioObsLejos = new SpinnerNumberModel(Boid.getRadioObstaculoLejos(),0,1000,0.1);
//	JSpinner spinnerRadioObsLejos = new JSpinner(spRadioObsLejos);
	JLabel etiquetaVelMax = new JLabel("Maximun velocity");
	SpinnerNumberModel spVelMax = new SpinnerNumberModel(Boid.getVelMax(),0,100,1);
	JSpinner spinnerVelMax = new JSpinner(spVelMax);
	JLabel etiquetaNumBoids = new JLabel("Cuantity of boids");
	SpinnerNumberModel spNumBoids = new SpinnerNumberModel(20,1,200,1);
	JSpinner spinnerNumBoids = new JSpinner(spNumBoids);
	JButton pausa = new JButton("Play");
	JButton pausaObs = new JButton("Play obst");
	JButton colocarObs = new JButton("Place obstacles");
	JButton colocarBan = new JButton("Place flock");
	JMenuItem botonSalvar = new JMenuItem("Save scenary");
	JMenuItem botonCargar = new JMenuItem("Load scenary");
	JMenuItem botonBorrarBandada = new JMenuItem("Erase flock");
	JMenuItem botonCrearBandada = new JMenuItem("Create flock");
	JCheckBox checkBoxPintar = new JCheckBox("Draw");
	JButton configurar = new JButton("Configure batch simulation");
	JButton simulacionBatch = new JButton("Execute batch simulation");
//	JLabel tiempoConsumido = new JLabel("0");
	public double tiempo;
	ConfigParam configurador;
	
	public void init(){
		Container cp = getContentPane();
		//AÃ±adimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		menuBandada.add(colocarBan);
		barraMenu.add(menuBandada);
		//AÃ±adimos elementos al menu de archivo
		menuArchivo.add(botonCargar);		
		menuArchivo.add(botonSalvar);		
		barraMenu.add(menuArchivo);
//		AÃ±adimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		barraMenu.add(menuBandada);
		pintor = new Dibujante2();
		sim = new Simulador();
//		sim.getCp().iniciaNavega();
		pintor.introducirBandada(getSim().getBandada());
		pintor.addMouseListener(this);
		cp.add(pintor);
		this.setJMenuBar(barraMenu);
		JPanel panelSur = new JPanel(new FlowLayout());
		JPanel panelNorte = new JPanel(new FlowLayout());
//		panelSur.add(etiquetaPesoLider);
//		panelSur.add(spinnerPesoLider);
		panelSur.add(etiquetaVelReprod);
		panelSur.add(spinnerVelReprod);
		panelSur.add(etiquetaCohesion);
		panelSur.add(spinnerCohesion);
		panelSur.add(etiquetaSeparacion);
		panelSur.add(spinnerSeparacion);
		panelSur.add(etiquetaAlineacion);
		panelSur.add(spinnerAlineacion);
//		panelSur.add(etiquetaObjetivo);
//		panelSur.add(spinnerObjetivo);
//		panelSur.add(etiquetaEvitaObs);
//		panelSur.add(spinnerEvitaObs);
//		panelSur.add(etiquetaObsCerca);
//		panelSur.add(spinnerEvitaObsCerca);
//		panelSur.add(etiquetaCompLateral);
//		panelSur.add(spinnerCompLateral);
		panelSur.add(etiquetaRadObs);
		panelSur.add(spinnerRadioObs);
//		panelSur.add(etiquetaRadObsLejos);
//		panelSur.add(spinnerRadioObsLejos);
		panelSur.add(etiquetaVelMax);
		panelSur.add(spinnerVelMax);
		panelNorte.add(checkBoxPintar);
		panelNorte.add(etiquetaNumBoids);
		panelNorte.add(spinnerNumBoids);
//		panelNorte.add(botonCrearBandada);
//		panelNorte.add(botonBorrarBandada);
		panelNorte.add(pausa);
		panelNorte.add(pausaObs);
		panelNorte.add(colocarObs);
		panelNorte.add(colocarBan);
		panelNorte.add(configurar);
		panelNorte.add(simulacionBatch);
//		panelNorte.add(tiempoConsumido);
//		panelNorte.add(botonSalvar);
//		panelNorte.add(botonCargar);		
//		spinnerPesoLider.addChangeListener(this);
		spinnerVelReprod.addChangeListener(this);
		spinnerCohesion.addChangeListener(this);
		spinnerSeparacion.addChangeListener(this);
		spinnerAlineacion.addChangeListener(this);
//		spinnerObjetivo.addChangeListener(this);
//		spinnerEvitaObs.addChangeListener(this);
		spinnerRadioObs.addChangeListener(this);
//		spinnerCompLateral.addChangeListener(this);
//		spinnerRadioObsCerca.addChangeListener(this);
//		spinnerRadioObsLejos.addChangeListener(this);
		spinnerVelMax.addChangeListener(this);
		spinnerNumBoids.addChangeListener(this);
		pausa.addActionListener(this);
		pausaObs.addActionListener(this);
		colocarObs.addActionListener(this);
		colocarBan.addActionListener(this);
		botonSalvar.addActionListener(this);
		botonCargar.addActionListener(this);
		botonBorrarBandada.addActionListener(this);
		botonCrearBandada.addActionListener(this);
		checkBoxPintar.addActionListener(this);
		configurar.addActionListener(this);
		simulacionBatch.addActionListener(this);
		cp.add(BorderLayout.SOUTH,panelSur);
		cp.add(BorderLayout.NORTH,panelNorte);
		// Creamos el simulador
//		sim = new Simulador();
	}
	
	public void salvarEscenario(String fichero,Simulador sim){
		try {
			File file = new File(fichero);
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
			oos.writeObject(sim.getObstaculos());
			oos.close();
		} catch (IOException ioe) {
			System.err.println("Error al escribir en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	
	public void cargarEscenario(String fichero,Simulador sim) throws ClassNotFoundException{
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			sim.setObstaculos((Vector<Obstaculo>) ois.readObject());
//			pintor.obstaculosPintar = sim.getObstaculos();
//			repaint();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al leer en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	
	public void stateChanged(ChangeEvent e) {
//		if (e.getSource() == spinnerPesoLider){
//			Boid.setPesoLider(spPesoLider.getNumber().doubleValue());
//		}
		if (e.getSource() == spinnerVelReprod){
			setVelReprod(spVelReprod.getNumber().longValue());
		}
		if (e.getSource() == spinnerCohesion){
			Boid.setPesoCohesion(spCohesion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerSeparacion){
			Boid.setPesoSeparacion(spSeparacion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerAlineacion){
			Boid.setPesoAlineacion(spAlineacion.getNumber().doubleValue());
		}
//		if (e.getSource() == spinnerObjetivo){
//			Boid.setPesoObjetivo(spObjetivo.getNumber().doubleValue());
//		}
//		if (e.getSource() == spinnerEvitaObs){
//			Boid.setPesoObstaculo(spEvitaObs.getNumber().doubleValue());
//		}
//		if (e.getSource() == spinnerCompLateral){
//			Boid.setPesoCompensacionLateral(spCompLateral.getNumber().doubleValue());
//		}
		if (e.getSource() == spinnerRadioObs){
			Boid.setRadioObstaculo(spRadioObs.getNumber().doubleValue());
		}
//		if (e.getSource() == spinnerRadioObsLejos){
//			Boid.setRadioObstaculoLejos(spRadioObsLejos.getNumber().doubleValue());
//		}
		if (e.getSource() == spinnerVelMax){
			Boid.setVelMax(spVelMax.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerNumBoids){
			this.getSim().setTamanoBandada(spNumBoids.getNumber().intValue());
		}
	}

	public void actionPerformed(ActionEvent e){
		if (e.getSource() == pausa){
			if (!play){ // La etiqueta del botÃ³n cambia
				pausa.setText("Pause");
				tiempo = System.currentTimeMillis();
				botonBorrarBandada.setEnabled(false);
				botonCrearBandada.setEnabled(false);
				setObjetivoEncontrado(false);
			}
			if (play){
				pausa.setText("Play");
				botonBorrarBandada.setEnabled(true);
				botonCrearBandada.setEnabled(true);
			}
			play = !play;
		}
		if (e.getSource() == pausaObs){
			if (!playObs){ // La etiqueta del botÃ³n cambia
				pausaObs.setText("Pause Obs");
				tiempo = System.currentTimeMillis();
				botonBorrarBandada.setEnabled(false);
				botonCrearBandada.setEnabled(false);
				setObjetivoEncontrado(false);
			}
			if (playObs){
				pausaObs.setText("Play Obs");
				botonBorrarBandada.setEnabled(true);
				botonCrearBandada.setEnabled(true);
			}
			playObs = !playObs;
		}
		
		if (e.getSource() == colocarObs){
			if (!colocandoObs){ // La etiqueta del botÃ³n cambia
				colocarObs.setText("Colocando obstÃ¡culos");				
				colocarBan.setEnabled(false);
			}
			if (colocandoObs){
				colocarObs.setText("Colocar obstÃ¡culos");
				colocarBan.setEnabled(true);
			}
			colocandoObs = !colocandoObs;
		}
		if (e.getSource() == colocarBan){			
			if (!colocandoBan){ // La etiqueta del botÃ³n cambia
				colocarBan.setText("Colocando Bandada");
				colocarObs.setEnabled(false);
			}
			if (colocandoBan){
				colocarBan.setText("Colocar la Bandada");
				colocarObs.setEnabled(true);
			}
			colocandoBan = !colocandoBan;
		}
        if (e.getSource() == botonSalvar) {            
            int devuelto = selectorArchivo.showSaveDialog(null);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = selectorArchivo.getSelectedFile();
                System.out.println(file.getAbsolutePath());
                salvarEscenario(file.getAbsolutePath(),this.getSim());
            }            
        }
        if (e.getSource() == botonCargar) {            
            int devuelto = selectorArchivo.showOpenDialog(null);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = selectorArchivo.getSelectedFile();               
                try {
					cargarEscenario(file.getAbsolutePath(),this.getSim());
				} catch (ClassNotFoundException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
                
            }            
        }
        if (e.getSource() == botonBorrarBandada){
        	this.getSim().borrarBandada();
        	pintor.getBandadaPintar().clear();
        	pintor.repaint();
        }
        if (e.getSource() == botonCrearBandada){
//        	this.getSim().borrarBandsetTamanoBandadaada();
        	pintor.getBandadaPintar().clear();
        	this.getSim().crearBandada(spNumBoids.getNumber().intValue());
        }
        if (e.getSource() == checkBoxPintar){
        	pintarEscena = checkBoxPintar.isSelected();
        }
        if (e.getSource() == configurar){
        	if (configurador == null){
        		configurador = new ConfigParam();
        		Console.run(configurador,1000,300);
        	}        	        	
        }
        if (e.getSource() == simulacionBatch){
        	if(!batch){        		
        		if (getConfigurador().getVectorSim().size() > 0){
            		System.out.println("Existe un vector de puntos de diseño de tamaño "+getConfigurador().getVectorSim().size());
            		batch = true;
            		simulacionBatch.setText("Parar simulaciÃ³n batch");
            		colocarBan.setEnabled(false);
    				colocarObs.setEnabled(false);
    				pausa.setEnabled(false);
//            		simulacionBatch.setEnabled(false);
            	}
        	}
        	else{
        		simulacionBatch.setText("Ejecutar simulacion batch");
        		colocarBan.setEnabled(true);
				colocarObs.setEnabled(true);
				pausa.setEnabled(true);
				batch = false;
        	}
        	        	
        }
	}
	
	public void mouseClicked(MouseEvent e) {
		if (colocandoObs){
			Point2D posicionReal = pintor.pixel2Point(e.getX(),e.getY());
			double pos[] = {posicionReal.getX(),posicionReal.getY()};
//			System.out.println("posiciÃ³n real del obstÃ¡culo "+posicionReal.getX()+" "+
//					posicionReal.getY());
			double vel[] = {0,0};
			double rumbo[] = {0,0};
			Matrix posicion = new Matrix(pos,2);			
			int i = 0;
			if(getSim().getObstaculos().size()%2 <= 0){
				vel[0] = 0;
//				vel[1] = 2;
				vel[1] = 0;
				rumbo[0] = 0;
//				rumbo[1] = 2;
				rumbo[1] = 0;
			}else{
				vel[0] = 0;
//				vel[1] = -2;
				vel[1] = 0;
				rumbo[0] = 0;
//				rumbo[1] = -2;
				rumbo[1] = 0;
			}
			Matrix velocidad = new Matrix(vel,2);
			Matrix rumboDes = new Matrix(rumbo,2);
			Obstaculo nuevoObs = new Obstaculo(posicion,velocidad,rumboDes);
			this.getSim().getObstaculos().add(nuevoObs);
			pintor.introducirObstaculo(nuevoObs);
//			System.out.println("Hay "+pintor.obstaculosPintar.size()+" obstÃ¡culos");
			repaint();
		}
		if (colocandoBan){
			Point2D posIniReal = pintor.pixel2Point(e.getX(),e.getY());
			double pos[] = {posIniReal.getX(),posIniReal.getY()};
			Matrix posicion = new Matrix(pos,2);
			this.getSim().posicionarBandada(posicion);
			this.getSim().setPosInicial(posicion);
			pintor.setPosCoche(posicion);
//			System.out.println(getSim().getBandada());
			pintor.introducirBandada(this.getSim().getBandada());
			repaint();
		}
		if (!colocandoBan && !colocandoObs){		
			Point2D objetivo = pintor.pixel2Point(e.getX(),e.getY());
			Boid.setObjetivo(objetivo.getX(),objetivo.getY());
			this.getSim().setObjetivo(objetivo.getX(),objetivo.getY());// cambia las coordenadas del objetivo
			repaint();
		}
		
	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated methodsetTamanoBandada stub
		
	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public double getTiempo() {
		return tiempo;
	}

	public void setTiempo(double tiempo) {
		this.tiempo = tiempo;
	}

	public boolean isObjetivoEncontrado() {
		return objetivoEncontrado;
	}

	public void setObjetivoEncontrado(boolean objetivoEncontrado) {
		this.objetivoEncontrado = objetivoEncontrado;
	}

	public Simulador getSim() {
		return sim;
	}

	public void setSim(Simulador sim) {
		this.sim = sim;
	}

	public ConfigParam getConfigurador() {
		return configurador;
	}

	public Dibujante2 getPintor() {
		return pintor;
	}

	public void setPintor(Dibujante2 pintor) {
		this.pintor = pintor;
	}
	public static void main(String[] args){
		//-------------------------------------------------------------------------------------------
		//---------Creamos la ventana y definimos el tamaño del escenario----------------------------
		//-------------------------------------------------------------------------------------------
		MuestraBoids gui = new MuestraBoids();
		Console.run(gui,1200,500);
		int alturaPanel = gui.pintor.getHeight();
		int anchuraPanel = gui.pintor.getWidth();
		double longitudEscenario = 100; // inicialmente a 60
		double anchuraEscenario = longitudEscenario*alturaPanel/anchuraPanel;//30
		gui.getSim().setAnchoEscenario(anchuraEscenario);
		gui.getSim().setLargoEscenario(longitudEscenario);
		gui.getPintor().setEsqInferiorDerecha(longitudEscenario,0);
		gui.getPintor().setEsqSuperiorIzquierda(0,longitudEscenario*alturaPanel/anchuraPanel);
		
		//------------Defino las variables para almacenar los datos para la estadística-------------
		//------------------------------------------------------------------------------------------
		
		Vector<Matrix> vectorPosCoche = new Vector<Matrix>();
		Vector<Double> yawCoche = new Vector<Double>();
		Vector<Double> distMinObst = new Vector<Double>();
		double tiempoLlegadaCoche = 0;
		double tiempoLlegadaCocheAEstrella = 0;
		double tiempoLlegadaCocheSolitario = 0;
		
		Vector<Matrix> vectorPosCocheAEstrella = new Vector<Matrix>();
		Vector<Double> yawCocheAEstrella = new Vector<Double>();
		Vector<Double> distMinObstAEstrella = new Vector<Double>();
		
		Vector<Matrix> vectorPosCocheSolitario = new Vector<Matrix>();
		Vector<Double> yawCocheSolitario = new Vector<Double>();
		Vector<Double> distMinObstSolitario = new Vector<Double>();
		
		//-------------------------------------------------------------------------------------------
		//---------Defino las condiciones de la simulación, número de obstáculos,etc-----------------
		//-------------------------------------------------------------------------------------------
		int numSimu = 0;
		int simuDeseadas = 300;
//		int numObstaculos = 20; // Los obstáculos se generan dentro del bucle principal usando el método generaobstaculosEquiespaciadosCruce() de la clase simulador
		double sepEntreObst = 5;
		double distCercana = 3;		
		gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());
		int indMinAnt = 0;
		double tMaximo = 100; // 80,120
		double distCocheObjetivo = Double.POSITIVE_INFINITY;
		double distCocheAEstrellaObjetivo = Double.POSITIVE_INFINITY;
		double distCocheSolitarioObjetivo  = Double.POSITIVE_INFINITY;
		
		boolean cocheEnObjetivo = false;
		boolean cocheAEstrellaEnObjetivo = false;
		boolean cocheSolitarioEnObjetivo = false;
		
		int simCompletasCoche = 0;
		int simCompletasCocheAEstrella = 0;
		int simCompletasCocheSolitario = 0;
		
		//--------------------------------------------------------------------------------------------
		//----------Indico la posición del objetivo y la posición inicial-----------------------------
		//--------------------------------------------------------------------------------------------
		double[] objetivo = {longitudEscenario-3,anchuraEscenario/2};
		gui.getSim().setObjetivo(new Matrix(objetivo,2));
		double[] inicial = {3,anchuraEscenario/2};
		gui.getSim().setPosInicial(new Matrix(inicial,2));
//		gui.pintor.setPosCoche(gui.getSim().getPosInicial());
//		gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
		
		//-------------------------------------------------------------------
		//----------------BUCLE PRINCIPAL------------------------------------
		//-------------------------------------------------------------------
		
		
//		while (true){
		LoggerFactory.activaLoggers(1000);
		System.out.println("Antes del while");
//		gui.getSim().creaRejilla(0.5);
		int contConfigurador = 0;
		while (numSimu < simuDeseadas){
			System.out.println("Despues del while");
			if (gui.play){
				//Si existe vector de configuración aplicamos las variaciones de los parámetros 
				//para cada simulación
//				if (gui.getConfigurador().getVectorSim().size() != 0){
//					simuDeseadas = gui.getConfigurador().getVectorSim().size();
//					System.out.println("numero de simulaciones que se van a realizar "+simuDeseadas);
//					gui.getSim().configurador(gui.getConfigurador().getVectorSim().elementAt(contConfigurador),
//    						gui.getConfigurador().getNomParam());
//				}
//				contConfigurador++;
				//------------------------------------------------------------------------------
				//------------reinicio todo para la siguiente simulación------------------------
				//------------------------------------------------------------------------------
				cocheEnObjetivo = false;
				cocheAEstrellaEnObjetivo = false;
				cocheSolitarioEnObjetivo = false;
				gui.getSim().setContIteraciones(0);
				vectorPosCoche.clear();
				vectorPosCocheAEstrella.clear();
				vectorPosCocheSolitario.clear();
				yawCoche.clear();
				yawCocheAEstrella.clear();
				yawCocheSolitario.clear();
				distMinObst.clear();
				distMinObstAEstrella.clear();
				distMinObstSolitario.clear();
				gui.pintor.eliminarObstaculos();
//				gui.getSim().generaObstaculos(numObstaculos,1.5);								
//				gui.getSim().generaObstaculosEquiespaciados(5, 1.5);
				gui.getSim().generaObstaculosEquiespaciadosCruce(sepEntreObst, 1.5,2.6,false);
				
				//------------------------------------------------------------------------------				
				//------------------leemos el escenario de un archivo---------------------------
				//------------------------------------------------------------------------------
				
//				try {
//					gui.cargarEscenario("C:/Users/Jesús/Dropbox/workspace/Octave/Boids/DatosBoids/escenario",gui.getSim());
//					System.out.println("cargamos el escenario");
//				} catch (ClassNotFoundException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
				gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());				
				gui.getSim().borrarBandada();
				gui.getSim().creaBandadaUniforme();
//				gui.getSim().crearBandada(20,gui.getSim().getContIteraciones());
//				gui.getSim().setPosInicial(new Matrix(inicial,2));
				
				gui.getSim().getModCoche().setContParadas(0);
				gui.getSim().getModeCocheAEstrella().setContParadas(0);
				gui.getSim().getCocheSolitario().setContParadas(0);				
				
				//Introducimos datos en la rejilla y la creamos	
				gui.getSim().creaRejilla(1);
				gui.getSim().getRejilla().setSim(gui.getSim());
				gui.getSim().getRejilla().setVelCoche(gui.getSim().getModeCocheAEstrella().getVelocidad());
				gui.getSim().getRejilla().setGoalPos(gui.getSim().getObjetivo().get(0, 0),gui.getSim().getObjetivo().get(1, 0));
				gui.getSim().getRejilla().setStartPos(gui.getSim().getModeCocheAEstrella().getX(),gui.getSim().getModeCocheAEstrella().getY());
//				System.out.println("rejilla sin crear");
//				gui.getSim().creaRejilla();
////				System.out.println("rejilla creada");
//				gui.getSim().getRejilla().setGoalPos(gui.getSim().getObjetivo().get(0, 0),gui.getSim().getObjetivo().get(0, 0));
//				gui.getSim().getRejilla().setStartPos(gui.getSim().getPosInicial().get(0, 0),gui.getSim().getPosInicial().get(1, 0));
				gui.pintor.setPosCoche(gui.getSim().getModCoche().getX(),gui.getSim().getModCoche().getY());
				gui.pintor.setPosCocheAEstrella(gui.getSim().getModeCocheAEstrella().getX(),gui.getSim().getModeCocheAEstrella().getY());
//				gui.pintor.setPosCocheSolitario(gui.getSim().getPosCocheSolitario());
				gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
				gui.pintor.setYawCocheAEstrella(gui.getSim().getModeCocheAEstrella().getYaw());
//				gui.pintor.setYawCocheSolitario((gui.getSim().getCocheSolitario().getYaw()));
				distCocheObjetivo = gui.getSim().getObjetivo().minus(gui.getSim().getPosInicial()).norm2();
				distCocheAEstrellaObjetivo = gui.getSim().getObjetivo().minus(gui.getSim().getPosInicial()).norm2();
//				distCocheSolitarioObjetivo = gui.getSim().getObjetivo().minus(gui.getSim().getPosInicial()).norm2();
				double tAnt = System.currentTimeMillis()/1000;
				double tSim = System.currentTimeMillis()/1000;		
				double tInicioIteracion = System.currentTimeMillis()/1000;
				
				//-----------La simulación acabará cuando todos los coches lleguen al objetivo o cuando haya-------------------
				//-----------transcurrido más tiempo del estipulado como bueno para una sola simulación---------------
				//-----------Para evitar que se atasque toda la simulación por lotes----------------------------------
				
//				while((distCocheObjetivo > distCercana)&&(tSim-tAnt < tMaximo)){
//				while(((!cocheEnObjetivo)||(!cocheAEstrellaEnObjetivo)||(!cocheSolitarioEnObjetivo))&&(tSim-tAnt < tMaximo)){
				while(((!cocheEnObjetivo)||(!cocheAEstrellaEnObjetivo))&&(tSim-tAnt < tMaximo)){
//				while((!cocheEnObjetivo)&&(tSim-tAnt < tMaximo)){
//				while((distCocheObjetivo - distCercana > 0)){
//					if ((distCocheObjetivo < distCercana)){
//						System.out.println("debería saltar a la siguiente simulación");
//						break;
//					}
						
					//Código para temporizar la reproducción de la simulación
					if(gui.play){
						try {
							Thread.sleep(gui.getVelReprod()-(long)(tSim-tInicioIteracion));
						} catch (Exception e) {
							System.out.println("No se pudo dormir el hilo principal de ejecución, el valor de velReprod es "+gui.getVelReprod());
						}
						tInicioIteracion = System.currentTimeMillis()/1000;
						if (gui.playObs){
							gui.getSim().moverObstaculos();
						}
						
//						gui.getSim().moverBoids(gui.getSim().getModCoche());
						gui.getSim().moverBoids(gui.getSim().getModeCocheAEstrella());

						//Introducimos datos en la rejilla y la creamos	

						gui.getSim().creaRejilla(0.3);
						gui.getSim().getRejilla().setSim(gui.getSim());
						gui.getSim().getRejilla().setVelCoche(gui.getSim().getModeCocheAEstrella().getVelocidad());
						gui.getSim().getRejilla().setGoalPos(gui.getSim().getObjetivo().get(0, 0),gui.getSim().getObjetivo().get(1, 0));
						gui.getSim().getRejilla().setStartPos(gui.getSim().getModeCocheAEstrella().getX(),gui.getSim().getModeCocheAEstrella().getY());

						//-----------------------------------------------------------------------
						//-----Cálculo de las diferentes trayectorias----------------------------
						//-----------------------------------------------------------------------
//						gui.getSim().setRutaDinamica(gui.getSim().calculaRutaDinamica(gui.getSim().getModCoche()));
//						gui.getSim().setRutaAEstrellaGrid(gui.getSim().suavizador(gui.getSim().getRejilla().busquedaAEstrellaConMarcado(gui.getSim().getUmbralCercania()),0.25));
					
						gui.getSim().setRutaAEstrellaGrid(gui.getSim().getRejilla().busquedaAEstrellaConMarcado(gui.getSim().getUmbralCercania()));

						//System.out.println(gui.getSim().getRutaAEstrellaGrid().size());
//						gui.setRutaDinamica(gui.getSim().calculaRutaDinamica(indice));
//						gui.setRutaDinamica(gui.getSim().calculaRutaDinamica());
						// Si la ruta no ha sido intersectada por un obstáculo no se vuelve a calcular
//						if (gui.getSim().compruebaRuta() || !gui.getSim().isRutaCompleta()){							
//							gui.getSim().setRutaDinamica(gui.getSim().calculaRutaDinamica(gui.getSim().getModCoche()));
//							gui.getSim().setRutaDinamica(gui.getSim().busquedaAEstrella(gui.getSim().getModCoche()));
//							gui.getSim().busquedaAEstrella();
//							gui.getSim().setRutaDinamica(gui.getSim().busquedaAEstrella());
//							gui.getSim().setRutaDinamica(gui.getSim().getRejilla().busquedaAEstrella());
//							if (gui.getSim().getRutaDinamica().size() <= 3){
//////								gui.setRutaDinamica(gui.getSim().calculaRutaDinamica());
////								gui.setRutaDinamica(gui.getSim().busquedaAEstrella());
//////								gui.setRutaDinamica(gui.getSim().busquedaAEstrella());
//							}else{
//								gui.getSim().setRutaDinamica(gui.getSim().suavizador(gui.getSim().getRutaDinamica(),0.25));
//								gui.setRutaDinamica(gui.getSim().suavizador(gui.getSim().getRutaDinamica(),0.25));
//							}
//						}					
//						gui.getSim().setRutaDinamica(gui.getSim().suavizador(0.1));
//						gui.pintor.setRutaDinamica(gui.getSim().busquedaAEstrella());
						gui.pintor.setRutaDinamica(gui.getSim().getRutaDinamica());
						gui.pintor.setRutaAEstrella(gui.getSim().getRutaAEstrellaGrid());
//						gui.pintor.setRutaDinamica(gui.getRutaDinamica());
//						gui.pintor.setHorPrediccion(gui.getSim().getContPred().getHorPrediccion());
//						gui.pintor.setPrediccion(gui.getSim().getContPred().getPrediccionPosPorFilas());
						gui.pintor.setHorPrediccion(gui.getSim().getContPredAEstrella().getHorPrediccion());
						gui.pintor.setPrediccion(gui.getSim().getContPredAEstrella().getPrediccionPosPorFilas());
						//Arreglar este seteo salvaje
						gui.getSim().setRutaParcial(true);
						
						//-----------Movemos los vehículos---------------------------------
						//-----------------------------------------------------------------
						  
//						gui.getSim().moverPtoInicial(tAnt, gui.getSim().getTs());
//						if(!cocheEnObjetivo){
//							gui.getSim().moverVehiculo(gui.getSim().getModCoche(),gui.getSim().getRutaDinamica(),
//									 gui.getSim().getTs(),false,true,gui.getSim().getContPred());
//						}
						if(!cocheAEstrellaEnObjetivo){
//							System.out.println("el coche no está en el objetivo");
							gui.getSim().moverVehiculo(gui.getSim().getModeCocheAEstrella(),gui.getSim().getRutaAEstrellaGrid(),
									 gui.getSim().getTs(),true,false,gui.getSim().getContPredAEstrella());
						}
//						if(!cocheSolitarioEnObjetivo){
//							gui.pintor.setVectorDirectorCoche(gui.getSim().moverCocheSolitario(gui.getSim().getTs()));
//						}
						
						//-----Medimos la distancia a la que se encuentran los coches del objetivo----------
						//----------------------------------------------------------------------------------
						
						double posCocheBoids[] = {gui.getSim().getModCoche().getX(),gui.getSim().getModCoche().getY()};
						Matrix posiCocheBoids = new Matrix(posCocheBoids,2);
						distCocheObjetivo = gui.getSim().getObjetivo().minus(posiCocheBoids).norm2();

						double posCocheAEstrella[] = {gui.getSim().getModeCocheAEstrella().getX(),gui.getSim().getModeCocheAEstrella().getY()};
						Matrix posiCocheAEstrella = new Matrix(posCocheAEstrella,2);
						distCocheAEstrellaObjetivo = gui.getSim().getObjetivo().minus(posiCocheAEstrella).norm2();
//						
//						double posCocheSolitario[] = {gui.getSim().getCocheSolitario().getX(),gui.getSim().getCocheSolitario().getY()};
//						Matrix posiCocheSolitario = new Matrix(posCocheSolitario,2);
//						distCocheSolitarioObjetivo = gui.getSim().getObjetivo().minus(posiCocheSolitario).norm2();
						
						//---------Marcamos que vehículos han alcanzado el objetivo-------------------------------
						//----------------------------------------------------------------------------------------
						
						if (distCocheObjetivo < distCercana){
							cocheEnObjetivo = true;							
						}
						if (distCocheAEstrellaObjetivo < distCercana){
							cocheAEstrellaEnObjetivo = true;
						}
//						if (distCocheSolitarioObjetivo < distCercana){
//							cocheSolitarioEnObjetivo = true;
//						}
						
						//---------Rellenamos los datos de los caminos seguidos por los coches--------------
						//----------------------------------------------------------------------------------
//						if(!cocheEnObjetivo){
//							double[] posCocheAux = {gui.getSim().getModCoche().getX(),
//									gui.getSim().getModCoche().getY()};
//							vectorPosCoche.add(new Matrix(posCocheAux,2));
//							yawCoche.add(gui.getSim().getModCoche().getYaw());
//							distMinObst.add(gui.getSim().distObstaculoMasCercanoAlCoche(gui.getSim().getModCoche()));
//						}
						
//						if(!cocheAEstrellaEnObjetivo){
//							double[] posCocheAEstrellaAux = {gui.getSim().getModeCocheAEstrella().getX(),
//									gui.getSim().getModeCocheAEstrella().getY()};
//							vectorPosCocheAEstrella.add(new Matrix(posCocheAEstrellaAux,2));
//							yawCocheAEstrella.add(gui.getSim().getModeCocheAEstrella().getYaw());
//							distMinObstAEstrella.add(gui.getSim().distObstaculoMasCercanoAlCoche(gui.getSim().getModeCocheAEstrella()));
//						}						
//						
//						if (!cocheSolitarioEnObjetivo){
//							double[] posCocheSolitarioAux = {gui.getSim().getCocheSolitario().getX(),
//									gui.getSim().getCocheSolitario().getY()};
//							vectorPosCocheSolitario.add(new Matrix(posCocheSolitarioAux,2));
//							yawCocheSolitario.add(gui.getSim().getCocheSolitario().getYaw());
//							distMinObstSolitario.add(gui.getSim().distObstaculoMasCercanoAlCoche(gui.getSim().getCocheSolitario()));
//						}						
						
//						System.out.println("la distancia del coche hasta el objetivo es "+distCocheObjetivo);
						
						//-----------Pasamos las variables para la representación gráfica-------------------
						//----------------------------------------------------------------------------------
						
						gui.pintor.setPosCoche(gui.getSim().getModCoche().getX(),gui.getSim().getModCoche().getY());
						gui.pintor.setPosCocheAEstrella(gui.getSim().getModeCocheAEstrella().getX(),gui.getSim().getModeCocheAEstrella().getY());
//						gui.pintor.setPosCocheSolitario(gui.getSim().getPosCocheSolitario());
						gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
						gui.pintor.setYawCocheAEstrella(gui.getSim().getModeCocheAEstrella().getYaw());
						gui.pintor.setHorPrediccion(gui.getSim().getContPredAEstrella().getHorPrediccion());
						gui.pintor.setPrediccion(gui.getSim().getContPredAEstrella().getPrediccionPosPorFilas());
//						gui.pintor.setYawCocheSolitario((gui.getSim().getCocheSolitario().getYaw()));						
						/*if (gui.getRutaDinamica().size()>1){ //Comprobar si hay ruta que seguir					
							Trayectoria tr = new Trayectoria(gui.getSim().traduceRuta(gui.getRutaDinamica()));
							Trayectoria trMasPuntos = new Trayectoria(tr,0.1);
							trMasPuntos.situaCoche(gui.getSim().getModCoche().getX(),gui.getSim().getModCoche().getY());
							if (flagUnaVez){
								gui.getSim().getCp().setRuta(trMasPuntos);
								flagUnaVez = false;
							}					
							gui.getSim().moverPtoInicial(System.currentTimeMillis(),System.currentTimeMillis()-tAnt);
						}								
						gui.pintor.setPosCoche(gui.getSim().getPosInicial());
						gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());*/
//						gui.getSim().simuPorLotes();				
						if (gui.pintarEscena)
							gui.pintor.repaint();	
						//recogemos el tiempo que transcurre en cada simulación
						tSim = System.currentTimeMillis()/1000;
					}
				}
				//-----------------Aquí acaba cada simulación-----------------------------------
				//------------------------------------------------------------------------------
				
				System.out.println("acabó la simulación "+numSimu);
				numSimu++;

				//-----------------Calculo la diferencia en el tiempo de llegada entre el coche A estrella----
				//-----------------y el coche de los boids----------------------------------------------------
				if(cocheEnObjetivo){
					tiempoLlegadaCoche = vectorPosCoche.size()*gui.getSim().getTs();
					simCompletasCoche++;				
				}
//				if(cocheAEstrellaEnObjetivo){
//					tiempoLlegadaCocheAEstrella = vectorPosCocheAEstrella.size()*gui.getSim().getTs();
//					simCompletasCocheAEstrella++;
//				}	
//				if(cocheSolitarioEnObjetivo){
//					tiempoLlegadaCocheSolitario = vectorPosCocheSolitario.size()*gui.getSim().getTs();
//					simCompletasCocheSolitario++;
//				}
				
				
				gui.getSim().getLogTiemposdeLlegada().add(tiempoLlegadaCoche,0,0);
				gui.getSim().getLogParadas().add(gui.getSim().getModCoche().getContParadas(),0,0);
				gui.getSim().getLogParametrosBoid().add(Boid.getRadioSeparacion(),Boid.getRadioObstaculo());
				
//				gui.getSim().getLogTiemposdeLlegada().add(tiempoLlegadaCoche,tiempoLlegadaCocheAEstrella,tiempoLlegadaCocheSolitario);
//				gui.getSim().getLogParadas().add(gui.getSim().getModCoche().getContParadas(),
//												gui.getSim().getModeCocheAEstrella().getContParadas(),
//												gui.getSim().getCocheSolitario().getContParadas());
//				gui.getSim().getLogParametrosBoid().add(Boid.getRadioSeparacion(),Boid.getRadioObstaculo());
				

				//-----------------Calculamos la estadística para cada coche--------------------
				//------------------------------------------------------------------------------

				
				gui.getSim().calculaEstadistica(vectorPosCoche, yawCoche, distMinObst, gui.getSim().getTs(),
						gui.getSim().getLogEstadisticaCoche());
//				gui.getSim().calculaEstadistica(vectorPosCocheAEstrella, yawCocheAEstrella, distMinObstAEstrella,
//						gui.getSim().getTs(), gui.getSim().getLogEstadisticaCocheAEstrella());
//				gui.getSim().calculaEstadistica(vectorPosCocheSolitario, yawCocheSolitario, distMinObstSolitario,
//						gui.getSim().getTs(), gui.getSim().getLogEstadisticaCocheSolitario());
				
//				if (distCocheObjetivo <= distCercana){
////					CÃ¡lculo de las medias, varianzas,etc del estudio estadÃ­stico
//					double distEntrePtos = 0;
//					int numDatos = vectorPosCoche.size()-1;
//					double[] velCoche = new double[numDatos];
//					double[] acelCoche = new double[numDatos];
//					double sumaVel = 0;
//					double sumaYaw = 0;
//					double sumaAcel = 0;
//					double sumaDistMin = 0;
//					double mediaVel = 0;
//					double mediaYaw = 0;
//					double mediaAcel = 0;
//					double mediaDistMin = 0;
//					double varianzaVel = 0;
//					double varianzaYaw = 0;
//					double varianzaAcel = 0;
//					double varianzaDistMin = 0;
//					double sumaVarianzaVel = 0;
//					double sumaVarianzaYaw = 0;
//					double sumaVarianzaAcel = 0;
//					double sumaVarianzaDistMin = 0;
//					double desvTipicaVel = 0;
//					double desvTipicaYaw = 0;
//					double desvTipicaAcel = 0;
//					double desvtipicaDistMin = 0;
//					//cálculo de las medias
//					for (int h=0;h<numDatos;h++){
//						distEntrePtos = vectorPosCoche.elementAt(h+1).minus(vectorPosCoche.elementAt(h)).norm2();
//						sumaVel = sumaVel + distEntrePtos/gui.getSim().getTs();
//						velCoche[h] = distEntrePtos/gui.getSim().getTs();
//						sumaYaw = sumaYaw + yawCoche.elementAt(h);
//						sumaDistMin = sumaDistMin + distMinObst.elementAt(h);
//					}
//					for (int h=0;h<numDatos-1;h++){
//						acelCoche[h] = velCoche[h+1]-velCoche[h];
//						sumaAcel = sumaAcel + acelCoche[h];
//					}				
//					mediaVel = sumaVel/numDatos;
//					mediaYaw = sumaYaw/numDatos;
//					mediaAcel = sumaAcel/numDatos;
//					mediaDistMin = sumaDistMin/numDatos;
//					//calculo las varianzas
//					for (int k = 0;k<numDatos;k++){
//						sumaVarianzaVel = sumaVarianzaVel + Math.sqrt(Math.abs(velCoche[k]-mediaVel));
//						sumaVarianzaYaw = sumaVarianzaYaw + Math.sqrt(Math.abs(yawCoche.elementAt(k)-mediaYaw));
//						sumaVarianzaAcel = sumaVarianzaAcel + Math.sqrt(Math.abs(acelCoche[k]-mediaAcel));
//						sumaVarianzaDistMin = sumaVarianzaDistMin + Math.sqrt(Math.abs(distMinObst.elementAt(k)-mediaDistMin)); 
//					}
//					System.out.println("numDatos="+numDatos+" varianzaVel="+sumaVarianzaVel+" varianzaYaw="+sumaVarianzaYaw);
//					varianzaVel = sumaVarianzaVel/numDatos;
//					varianzaYaw = sumaVarianzaYaw/numDatos;
//					varianzaAcel = sumaVarianzaAcel/numDatos;
//					varianzaDistMin = sumaVarianzaDistMin/numDatos;
//					desvTipicaVel = Math.sqrt(varianzaVel);
//					desvTipicaYaw = Math.sqrt(varianzaYaw);
//					desvTipicaAcel = Math.sqrt(varianzaAcel);
//					desvtipicaDistMin = Math.sqrt(varianzaDistMin);
//					//rellenamos los valores en el loger estadístico
//					gui.getSim().getLogEstadisticaCoche().add(mediaVel,desvTipicaVel,mediaYaw,desvTipicaYaw,
//							mediaAcel,desvTipicaAcel,mediaDistMin,desvtipicaDistMin);
//					System.out.println("Acabó la simulación "+numSimu+" de "+simuDeseadas+" y calculó la estadística, los valores son "
//							+mediaVel+" "+desvTipicaVel+" "+mediaYaw+" "+desvTipicaYaw+" "+mediaAcel+" "+desvTipicaAcel);
//
//				}
//				else{
//					System.out.println("El coche no alcanzó el objetivo en el tiempo establecido");
//				}
			}
			if (gui.batch){
				gui.getSim().getBoidsOk().clear();
//				gui.getSim().setNumBoidsOkDeseados(gui.getConfigurador().getNumBoidsOk());
				gui.getSim().setTiempoMax(gui.getConfigurador().getTMax());
				for(int i=0;i<gui.getConfigurador().getVectorSim().size() && gui.batch;i++){
					//Configuramos los parámetros de la bandada
	    				gui.getSim().configurador(gui.getConfigurador().getVectorSim().elementAt(i),
	    						gui.getConfigurador().getNomParam());
//	    				gui.getSim().crearBandada();
	    				gui.getSim().creaBandadaUniforme();
	    				gui.getSim().posicionarBandada(gui.getSim().getPosInicial());	    				
	    				gui.getSim().simuPorLotes();
//	    				gui.tiempoConsumido.setText("TardÃ³ " + gui.getSim().getTiempoInvertido() + " sec");	    				
	    				if (gui.pintarEscena){
	    					for(int j=0;j<gui.getSim().getBoidsOk().size();j++){
/*OJO! aqui se mejora la ruta*/			/*gui.getSim().getBoidsOk().elementAt(j).setRutaBoid(
	    								gui.getSim().mejoraRuta(
	    								gui.getSim().getBoidsOk().elementAt(j).getRutaBoid()));
	    						gui.getSim().getBoidsOk().elementAt(j).setRutaBoid(
	    								gui.getSim().mejoraRuta(
	    								gui.getSim().getBoidsOk().elementAt(j).getRutaBoid())); */
	    						//El identificador del tipo de camino serÃ¡ el nÃºmero de vertices que tiene el
	    						//camino despuÃ©s de haber mejorado la ruta
	    						int identificador = gui.getSim().getBoidsOk().elementAt(j).getRutaBoid().size();
	    						//Compruebo si el vector de caminos estÃ¡ vacio
	    						if(gui.getSim().getCaminos().size()<=0){
	    							//Creo un tipo nuevo de camino y aÃ±ado un boid al nÂº de boids que han tomado
	    							//este camino
	    							gui.getSim().getCaminos().add(new TipoCamino(identificador,1,
	    									gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones(),
	    									gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta()));
	    						}else{
	    							//Bucle para buscar coincidencias entre los identificadores existentes
		    						//y el identificador de la ruta del boid actual
		    						boolean encontrado = false;
		    						for(int k=0;k<gui.getSim().getCaminos().size() && !encontrado;k++){
		    							if(gui.getSim().getCaminos().elementAt(k).getIdentificador() == identificador){
		    								encontrado = true;
		    								gui.getSim().getCaminos().elementAt(k).addFrecuencia();
		    								gui.getSim().getCaminos().elementAt(k).addCoste(
		    										gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones());
		    								gui.getSim().getCaminos().elementAt(k).addLongitud(
		    										gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta());
		    							}
		    						}
		    						if(!encontrado){
		    							gui.getSim().getCaminos().add(new TipoCamino(identificador,1,		    									
			    							gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones(),
			    							gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta()));
		    						}
	    						}	
	    					}
	    					//Imprimimos en consola el resultado
	    					for(int q=0;q<gui.getSim().getCaminos().size();q++){
	    						System.out.println("La frec del camino tipo "+
	    								(gui.getSim().getCaminos().elementAt(q).getIdentificador()-1) +
	    								" es "+ gui.getSim().getCaminos().elementAt(q).getFrecuencia() + 
	    								" ,tiene un coste de " + gui.getSim().getCaminos().elementAt(q).calculaCoste() + 
	    								" y una longitud de " + gui.getSim().getCaminos().elementAt(q).getLongitudMedia());
	    					}
	    					gui.pintor.introducirBandada(gui.getSim().getBoidsOk());
	    					gui.pintor.repaint();     	    				
	    				}	    				
//	    				gui.pintor.introducirBandada(gui.getSim().getBandada());
//	    			}        				
	    				System.out.println("Ya acabó el simuporlotes");
	    				System.out.println(gui.getSim().getNumBoidsOk());
	    		}
				gui.simulacionBatch.setText("Ejecutar simulación batch");
        		gui.colocarBan.setEnabled(true);
				gui.colocarObs.setEnabled(true);
				gui.pausa.setEnabled(true);
				gui.batch = false;
//				gui.simulacionBatch.setEnabled(true);				
			}			
		}
		gui.getSim().getLogSimlacionesCompletadas().add(100*(simCompletasCoche/numSimu),100*(simCompletasCocheAEstrella/numSimu),
				100*(simCompletasCocheSolitario/numSimu));
//		LoggerFactory.vuelcaLoggersMATv4("C:/Users/Jesús/Dropbox/workspace/Octave/Boids/DatosBoids/"+sepEntreObst+"deSepEntreObst");
		LoggerFactory.vuelcaLoggersMATv4("C:/Users/Jesús/Dropbox/workspace/Octave/Boids/DatosBoids/VariacionParametros");
		System.out.println("Acabaron todas las simulaciones");
	}
}