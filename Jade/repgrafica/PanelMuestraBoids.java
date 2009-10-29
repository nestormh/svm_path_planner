package repgrafica;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
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

import com.bruceeckel.swing.Console;

import Jama.Matrix;
import boids.Boid;
import boids.Obstaculo;

public class PanelMuestraBoids extends JApplet implements ChangeListener,ActionListener,MouseListener{

	private static double distOk = 50;
	private boolean objetivoEncontrado;
	private boolean play = false;
	private boolean colocandoObs = false;
	private boolean colocandoBan = false;
	private boolean pintarEscena = false;
	JFileChooser selectorArchivo = new JFileChooser(new File("./Escenarios"));
	// Esto no debería estar en la clase de la interfaz gráfica
	int tamanoBandada = 20;
	Vector<Boid> bandada = new Vector<Boid>();
	Vector<Boid> boidsOk = new Vector<Boid>();
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	//Hasta aqui
	JMenuBar barraMenu = new JMenuBar(); 
	JMenu menuArchivo = new JMenu("Archivo");
	JMenu menuBandada = new JMenu("Bandada");
	Dibujante pintor = new Dibujante();
	JLabel etiquetaPesoLider = new JLabel("Liderazgo");
	SpinnerNumberModel spPesoLider = new SpinnerNumberModel(Boid.getPesoLider(),0,100,0.1);
	JSpinner spinnerPesoLider = new JSpinner(spPesoLider);
	JLabel etiquetaCohesion = new JLabel("Cohesión");
	SpinnerNumberModel spCohesion = new SpinnerNumberModel(Boid.getPesoCohesion(),0,100,0.1);
	JSpinner spinnerCohesion = new JSpinner(spCohesion);
	JLabel etiquetaSeparacion = new JLabel("Separación");
	SpinnerNumberModel spSeparacion = new SpinnerNumberModel(Boid.getPesoSeparacion(),0,1000,0.1);
	JSpinner spinnerSeparacion = new JSpinner(spSeparacion);
	JLabel etiquetaAlineacion = new JLabel("Alineación");
	SpinnerNumberModel spAlineacion = new SpinnerNumberModel(Boid.getPesoAlineacion(),0,100,0.1);
	JSpinner spinnerAlineacion = new JSpinner(spAlineacion);
	JLabel etiquetaObjetivo = new JLabel("Velocidad Objetivo");
	SpinnerNumberModel spObjetivo = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,100,0.1);
	JSpinner spinnerObjetivo = new JSpinner(spObjetivo);
	JLabel etiquetaEvitaObs = new JLabel("Evitar Obstáculos");
	SpinnerNumberModel spEvitaObs = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
	JSpinner spinnerEvitaObs = new JSpinner(spEvitaObs);
	JLabel etiquetaVelMax = new JLabel("Velocidad Máxima");
	SpinnerNumberModel spVelMax = new SpinnerNumberModel(Boid.getVelMax(),0,100,1);
	JSpinner spinnerVelMax = new JSpinner(spVelMax);
	JLabel etiquetaNumBoids = new JLabel("Número de Boids");
	SpinnerNumberModel spNumBoids = new SpinnerNumberModel(this.getTamanoBan(),1,200,1);
	JSpinner spinnerNumBoids = new JSpinner(spNumBoids);
	JButton pausa = new JButton("Play");
	JButton colocarObs = new JButton("Colocar obstáculos");
	JButton colocarBan = new JButton("Colocar la bandada");
	JMenuItem botonSalvar = new JMenuItem("Salvar escenario");
	JMenuItem botonCargar = new JMenuItem("Cargar escenario");
	JMenuItem botonBorrarBandada = new JMenuItem("Borrar Bandada");
	JMenuItem botonCrearBandada = new JMenuItem("Crear Bandada");
	JCheckBox checkBoxPintar = new JCheckBox("Dibujar");
	public double tiempo;
	
	public void init(){
		Container cp = getContentPane();
		//Añadimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		menuBandada.add(colocarBan);
		barraMenu.add(menuBandada);
		//Añadimos elementos al menu de archivo
		menuArchivo.add(botonCargar);		
		menuArchivo.add(botonSalvar);		
		barraMenu.add(menuArchivo);
//		Añadimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		barraMenu.add(menuBandada);		
		pintor.addMouseListener(this);
		cp.add(pintor);
		this.setJMenuBar(barraMenu);
		JPanel panelSur = new JPanel(new FlowLayout());
		JPanel panelNorte = new JPanel(new FlowLayout());
		panelSur.add(etiquetaPesoLider);
		panelSur.add(spinnerPesoLider);
		panelSur.add(etiquetaCohesion);
		panelSur.add(spinnerCohesion);
		panelSur.add(etiquetaSeparacion);
		panelSur.add(spinnerSeparacion);
		panelSur.add(etiquetaAlineacion);
		panelSur.add(spinnerAlineacion);
		panelSur.add(etiquetaObjetivo);
		panelSur.add(spinnerObjetivo);
		panelSur.add(etiquetaEvitaObs);
		panelSur.add(spinnerEvitaObs);		
		panelSur.add(etiquetaVelMax);
		panelSur.add(spinnerVelMax);
		panelNorte.add(checkBoxPintar);
		panelNorte.add(etiquetaNumBoids);
		panelNorte.add(spinnerNumBoids);
//		panelNorte.add(botonCrearBandada);
//		panelNorte.add(botonBorrarBandada);
		panelNorte.add(pausa);
		panelNorte.add(colocarObs);
		panelNorte.add(colocarBan);
//		panelNorte.add(botonSalvar);
//		panelNorte.add(botonCargar);		
		spinnerPesoLider.addChangeListener(this);
		spinnerCohesion.addChangeListener(this);
		spinnerSeparacion.addChangeListener(this);
		spinnerAlineacion.addChangeListener(this);
		spinnerObjetivo.addChangeListener(this);
		spinnerEvitaObs.addChangeListener(this);
		spinnerVelMax.addChangeListener(this);
		spinnerNumBoids.addChangeListener(this);
		pausa.addActionListener(this);
		colocarObs.addActionListener(this);
		colocarBan.addActionListener(this);
		botonSalvar.addActionListener(this);
		botonCargar.addActionListener(this);
		botonBorrarBandada.addActionListener(this);
		botonCrearBandada.addActionListener(this);
		checkBoxPintar.addActionListener(this);
		cp.add(BorderLayout.SOUTH,panelSur);
		cp.add(BorderLayout.NORTH,panelNorte);
	}
	
	public Vector<Boid> getBandada(){
		return this.bandada;
	}
	
	public Vector<Obstaculo> getObstaculos(){
		return this.obstaculos;
	}
	
	public void setTamanoBan(int tamano){
		this.tamanoBandada = tamano;
	}
	
	public int getTamanoBan(){
		return this.tamanoBandada;
	}
	
	public void salvarEscenario(String fichero){
		try {
			File file = new File(fichero);
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
			oos.writeObject(obstaculos);
			oos.close();
		} catch (IOException ioe) {
			System.err.println("Error al escribir en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	
	public void cargarEscenario(String fichero) throws ClassNotFoundException{
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			obstaculos = (Vector<Obstaculo>) ois.readObject();
			pintor.obstaculosPintar = obstaculos;
			repaint();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al leer en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	public void traspasarBoid(int indBoid){
		if(bandada.size()>0){
			boidsOk.add(bandada.remove(indBoid));
			setTamanoBandada(bandada.size());
		}
		else
			System.err.println("La bandada principal está vacía");
	}
	public static void main(String[] args){
		PanelMuestraBoids ventana = new PanelMuestraBoids();
		ventana.setTamanoBan(ventana.getTamanoBandada());
		//Bucle para crear la bandada
		for (int j = 0;j<ventana.getTamanoBan();j++){
			double posAux[] = {Math.random()*800,Math.random()};
			double velAux[] = {Math.random(),Math.random()};
			Matrix posi = new Matrix(posAux,2);
			Matrix vel = new Matrix(velAux,2);			
			ventana.getBandada().add(new Boid(posi,vel));				
		}
		
		
		ventana.pintor.introducirBandada(ventana.getBandada());
		Console.run(ventana,1200,1000);
		int alturaPanel = ventana.pintor.getHeight();
		int anchuraPanel = ventana.pintor.getWidth();
//		double esquinaSupIzq[] = {0,0};
//		double esquinaInfIzq[] = {0,alturaPanel-20};
//		double esquinaSupDer[] = {anchuraPanel-10,0};
// 		Matrix matEsquinaSupIzq = new Matrix(esquinaSupIzq,2);
// 		Matrix matEsquinaInfIzq = new Matrix(esquinaInfIzq,2);
// 		Matrix matEsquinaSupDer = new Matrix(esquinaSupDer,2);
//		Obstaculo bordeIzquierdo = new Obstaculo(matEsquinaSupIzq,new Matrix(2,1));
//		bordeIzquierdo.setHeight(alturaPanel);
//		bordeIzquierdo.setWidth(10);
//		Obstaculo bordeSuperior = new Obstaculo(matEsquinaSupIzq,new Matrix(2,1));
//		bordeSuperior.setHeight(10);
//		bordeSuperior.setWidth(anchuraPanel);
//		Obstaculo bordeInferior = new Obstaculo(matEsquinaInfIzq,new Matrix(2,1));
//		bordeInferior.setHeight(10);
//		bordeInferior.setWidth(anchuraPanel);
//		Obstaculo bordeDerecho = new Obstaculo(matEsquinaSupDer,new Matrix(2,1));
//		bordeDerecho.setHeight(alturaPanel);
//		bordeDerecho.setWidth(10);
//		ventana.getObstaculos().add(bordeIzquierdo);
//		ventana.getObstaculos().add(bordeSuperior);
//		ventana.getObstaculos().add(bordeInferior);
//		ventana.getObstaculos().add(bordeDerecho);
		
//		Bucle para crear los bordes
		int cont = 1;
		for (int k=0; k<2;k++){			
			for (int i = 0;i < alturaPanel;i=i+20){
				double posObstaculos[] = {cont,i};
				double velObstaculos[] = {0,0};				
				Matrix posiObs = new Matrix(posObstaculos,2);
				Matrix velObs = new Matrix(velObstaculos,2);
				ventana.getObstaculos().add(new Obstaculo(posiObs,velObs));
			}
			cont = anchuraPanel-10;
		}
		cont = 1;
		for (int k=0; k<2;k++){			
			for (int i = 0;i < anchuraPanel;i=i+20){
				double posObstaculos[] = {i,cont};
				double velObstaculos[] = {0,0};				
				Matrix posiObs = new Matrix(posObstaculos,2);
				Matrix velObs = new Matrix(velObstaculos,2);
				ventana.getObstaculos().add(new Obstaculo(posiObs,velObs));
			}
			cont = alturaPanel-15;
		}
		ventana.pintor.introducirObstaculos(ventana.getObstaculos());
		double distMin = Double.POSITIVE_INFINITY;
		int indMin = 0;
		int indMinAnt = 0;
		boolean liderEncontrado = false;
		double distObjetivo = Double.POSITIVE_INFINITY;
		
		//----------------BUCLE PRINCIPAL------------------------------------		
		while (true){
//			&& (distObjetivo > 50)
//			!ventana.isObjetivoEncontrado() && 
			if (ventana.play){
				while(!ventana.isObjetivoEncontrado() && ventana.play && ventana.getBandada().size() > 0 ){
					for (int j = 0;j<ventana.getTamanoBan();j++){
						ventana.getBandada().elementAt(j).mover(ventana.getBandada()
								,ventana.getObstaculos(),j,Boid.getObjetivo());
						// Buscamos al lider
						if (ventana.getBandada().elementAt(j).isCaminoLibre()){
							double dist = ventana.getBandada().elementAt(j).getDistObjetivo();
							if (dist < distMin){
								distMin = dist;
								indMin = j;
								liderEncontrado = true;
							}
						}					
					}
					if (indMinAnt<ventana.getBandada().size())
						ventana.getBandada().elementAt(indMinAnt).setLider(false);
					if (liderEncontrado && (indMin<ventana.getBandada().size())){
						ventana.getBandada().elementAt(indMin).setLider(true);
						distObjetivo = Boid.getObjetivo().minus(ventana.getBandada().elementAt(indMin).getPosicion()).norm2();
					}
//					ventana.getBandada().elementAt(indMinAnt).setLider(false);
//					if (liderEncontrado){
//						ventana.getBandada().elementAt(indMin).setLider(true);
//						distObjetivo = Boid.getObjetivo().minus(ventana.getBandada().elementAt(indMin).getPosicion()).norm2();
////						ventana.setObjetivoEncontrado(distObjetivo < 15);
//					}
					if (distObjetivo < distOk ){
						ventana.traspasarBoid(indMin);
//						numBoidsOk++; // Incremento el numero de boids que han llegado al objetivo
					}
					else
						distObjetivo = Double.POSITIVE_INFINITY;
					indMinAnt = indMin;
					distMin = Double.POSITIVE_INFINITY;
					liderEncontrado = false;
					if (ventana.pintarEscena)
						ventana.pintor.repaint();
				}
				if (ventana.isObjetivoEncontrado()){
					System.out.println("El tiempo transcurrido es "+((System.currentTimeMillis()-ventana.getTiempo())/1000));
					ventana.pintor.repaint();					
//					System.out.println("El tiempo de repintado es "+((System.currentTimeMillis()-ventana.getTiempo())/1000));
				}
			}	
		}
	}
	

	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spinnerPesoLider){
			Boid.setPesoLider(spPesoLider.getNumber().doubleValue());
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
		if (e.getSource() == spinnerObjetivo){
			Boid.setPesoObjetivo(spObjetivo.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerEvitaObs){
			Boid.setPesoObstaculo(spEvitaObs.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerVelMax){
			Boid.setVelMax(spVelMax.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerNumBoids){
			this.setTamanoBan(spNumBoids.getNumber().intValue());
		}
	}

	public void actionPerformed(ActionEvent e){
		if (e.getSource() == pausa){
			if (!play){ // La etiqueta del botón cambia
				pausa.setText("Pausa");
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
		if (e.getSource() == colocarObs){
			if (!colocandoObs){ // La etiqueta del botón cambia
				colocarObs.setText("Colocando obstáculos");				
				colocarBan.setEnabled(false);
			}
			if (colocandoObs){
				colocarObs.setText("Colocar obstáculos");
				colocarBan.setEnabled(true);
			}
			colocandoObs = !colocandoObs;
		}
		if (e.getSource() == colocarBan){			
			if (!colocandoBan){ // La etiqueta del botón cambia
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
                salvarEscenario(file.getAbsolutePath());
            }            
        }
        if (e.getSource() == botonCargar) {            
            int devuelto = selectorArchivo.showOpenDialog(null);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = selectorArchivo.getSelectedFile();               
                try {
					cargarEscenario(file.getAbsolutePath());
				} catch (ClassNotFoundException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
                
            }            
        }
        if (e.getSource() == botonBorrarBandada){
        	getBandada().clear();
        	pintor.getBandadaPintar().clear();
        	pintor.repaint();
        }
        if (e.getSource() == botonCrearBandada){
        	getBandada().clear();
        	pintor.getBandadaPintar().clear();
        	for (int i=0; i< getTamanoBan();i++){
        		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1)));	
        	}
        }
        if (e.getSource() == checkBoxPintar){
        	pintarEscena = checkBoxPintar.isSelected();
        }
	}
	

	public void mouseClicked(MouseEvent e) {
		if (colocandoObs){
			double pos[] = {e.getX(),e.getY()};
			double vel[] = {0,0};
			Matrix posicion = new Matrix(pos,2);
			Matrix velocidad = new Matrix(vel,2);
			Obstaculo nuevoObs = new Obstaculo(posicion,velocidad);
			this.getObstaculos().add(nuevoObs);
			pintor.introducirObstaculo(nuevoObs);
			System.out.println("Hay "+pintor.obstaculosPintar.size()+" obstáculos");
			repaint();
		}
		if (colocandoBan){
			
			for (int i=0;i<this.getBandada().size();i++){
//				double pos[] = {e.getX()+Math.random()*getTamanoBan()*2, e.getY()+Math.random()*getTamanoBan()*2};
				double pos[] = {e.getX()+Math.random(), e.getY()+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {Math.random(),Math.random()};
				Matrix velo = new Matrix(vel,2);
				this.getBandada().elementAt(i).resetRuta();
				this.getBandada().elementAt(i).getForma().transform(AffineTransform.getTranslateInstance(pos[0]-getBandada().elementAt(i).getPosicion().get(0,0),
						pos[1]-getBandada().elementAt(i).getPosicion().get(1,0)));
				this.getBandada().elementAt(i).setPosicion(posi);			
				this.getBandada().elementAt(i).setVelocidad(velo);
			}
			pintor.introducirBandada(getBandada());
			repaint();
		}
		if (!colocandoBan && !colocandoObs){		
			Boid.setObjetivo(e.getX(),e.getY()); // cambia el coordenadas del objetivo
			repaint();
		}
		
	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		
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

	public int getTamanoBandada() {
		return tamanoBandada;
	}

	public void setTamanoBandada(int tamanoBandada) {
		this.tamanoBandada = tamanoBandada;
	}

	public boolean isObjetivoEncontrado() {
		return objetivoEncontrado;
	}

	public void setObjetivoEncontrado(boolean objetivoEncontrado) {
		this.objetivoEncontrado = objetivoEncontrado;
	}

}

class Dibujante extends JPanel{
	Vector<Boid> bandadaPintar = new Vector<Boid>();
	Vector<Obstaculo> obstaculosPintar = new Vector<Obstaculo>();
	
	public void introducirBoid(Boid b){
		bandadaPintar.add(b);		
	}
	
	public void introducirBandada(Vector<Boid> banda){
		for (int i =0;i<banda.size();i++){
			bandadaPintar.add(banda.elementAt(i));
		}
	}
	
	public void introducirObstaculo(Obstaculo b){
		obstaculosPintar.add(b);		
	}
	
	public void introducirObstaculos(Vector<Obstaculo> banda){
		for (int i =0;i<banda.size();i++){
			obstaculosPintar.add(banda.elementAt(i));
		}
	}
	
	public Vector<Boid> getBandadaPintar(){
		return bandadaPintar;
	}
	public Vector<Obstaculo> getObstaculoPintar(){
		return obstaculosPintar;
	}
	
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		Matrix centroMasa = new Matrix(2,1);
		super.paintComponent(g2);
		// Pinto los Boids
//		if (bandadaPintar.size() > 0){
			for (int i=0;i<bandadaPintar.size();i++){
				if (bandadaPintar.elementAt(i).isLider()){
					g2.setColor(Color.green);
					g2.drawOval((int)bandadaPintar.elementAt(i).getPosicion().get(0,0)-100,
							(int)bandadaPintar.elementAt(i).getPosicion().get(1,0)-100,
							200,200);
					GeneralPath rutaLider = new GeneralPath();
					rutaLider.moveTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(0,0),
							bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(1,0));
					for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
						rutaLider.lineTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0,0),
								bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1,0));
					}
					g2.draw(rutaLider);
				}
				else
					g2.setColor(Color.blue);
				g2.draw(bandadaPintar.elementAt(i).getForma());
				g2.fill(bandadaPintar.elementAt(i).getForma());
				g2.draw(bandadaPintar.elementAt(i).getLineaDireccion());
				GeneralPath ruta = new GeneralPath();
				ruta.moveTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(0,0),
				bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(1,0));
				for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
				ruta.lineTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0,0),
				bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1,0));
				}
				g2.draw(ruta);
				centroMasa = centroMasa.plus(bandadaPintar.elementAt(i).getPosicion());
			}
//		}
		// Pinto el centro de masa
		centroMasa.timesEquals((double)1/(double)bandadaPintar.size());		
		g2.setColor(Color.cyan);
		g2.drawOval((int)centroMasa.get(0,0)-2,(int)centroMasa.get(1,0)-2,4,4);
		
		// Pinto los obstáculos
		for (int i=0;i<obstaculosPintar.size();i++){
			g2.setColor(Color.red);
			g2.draw(obstaculosPintar.elementAt(i).getForma());
			g2.fill(obstaculosPintar.elementAt(i).getForma());	
			
		}
		// Pinto el objetivo
		g2.setColor(Color.magenta);
		g2.drawOval((int)Boid.getObjetivo().get(0,0),(int)Boid.getObjetivo().get(1,0),5,5);
	}
}
