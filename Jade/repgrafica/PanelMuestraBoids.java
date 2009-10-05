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
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
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

	private boolean play = false;
	private boolean colocandoObs = false;
	private boolean colocandoBan = false;
	private boolean salvar = false;
	private boolean cargar = false;
	JFileChooser selectorArchivo = new JFileChooser(new File("./Escenarios"));
	// Esto no debería estar en la clase de la interfaz gráfica
	int tamanoBandada;
	Vector<Boid> bandada = new Vector<Boid>();
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	//Hasta aqui
	Dibujante pintor = new Dibujante();
	JLabel etiquetaCohesion = new JLabel("Cohesión");
	SpinnerNumberModel spCohesion = new SpinnerNumberModel(Boid.getPesoCohesion(),0,100,0.01);
	JSpinner spinnerCohesion = new JSpinner(spCohesion);
	JLabel etiquetaSeparacion = new JLabel("Separación");
	SpinnerNumberModel spSeparacion = new SpinnerNumberModel(Boid.getPesoSeparacion(),0,100,0.01);
	JSpinner spinnerSeparacion = new JSpinner(spSeparacion);
	JLabel etiquetaAlineacion = new JLabel("Alineación");
	SpinnerNumberModel spAlineacion = new SpinnerNumberModel(Boid.getPesoAlineacion(),0,100,0.01);
	JSpinner spinnerAlineacion = new JSpinner(spAlineacion);
	JLabel etiquetaObjetivo = new JLabel("Velocidad Objetivo");
	SpinnerNumberModel spObjetivo = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,100,0.01);
	JSpinner spinnerObjetivo = new JSpinner(spObjetivo);
	JLabel etiquetaEvitaObs = new JLabel("Evitar Obstáculos");
	SpinnerNumberModel spEvitaObs = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,100,0.01);
	JSpinner spinnerEvitaObs = new JSpinner(spEvitaObs);
	JLabel etiquetaVelMax = new JLabel("Velocidad Máxima");
	SpinnerNumberModel spVelMax = new SpinnerNumberModel(Boid.getVelMax(),0,100,0.01);
	JSpinner spinnerVelMax = new JSpinner(spVelMax);
	JButton pausa = new JButton("Play");
	JButton colocarObs = new JButton("Pulse para colocar obstáculos");
	JButton colocarBan = new JButton("Pulse para colocar la bandada");
	JButton botonSalvar = new JButton("Salvar escenario");
	JButton botonCargar = new JButton("Cargar escenario");
	
	public void init(){
		//Esto no debería estar en la clase de la interfaz gráfica
//		bandada = new Vector<Boid>();
//		obstaculos = new Vector<Obstaculo>();
		// Hasta aqui
		Container cp = getContentPane();
//		cp.setLayout(new FlowLayout());
		pintor.addMouseListener(this);
		cp.add(pintor);
		JPanel panelSur = new JPanel(new FlowLayout());
		JPanel panelNorte = new JPanel(new FlowLayout());
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
		panelNorte.add(pausa);
		panelNorte.add(colocarObs);
		panelNorte.add(colocarBan);
		panelNorte.add(botonSalvar);
		panelNorte.add(botonCargar);
		spinnerCohesion.addChangeListener(this);
		spinnerSeparacion.addChangeListener(this);
		spinnerAlineacion.addChangeListener(this);
		spinnerObjetivo.addChangeListener(this);
		spinnerEvitaObs.addChangeListener(this);
		spinnerVelMax.addChangeListener(this);
		pausa.addActionListener(this);
		colocarObs.addActionListener(this);
		colocarBan.addActionListener(this);
		botonSalvar.addActionListener(this);
		botonCargar.addActionListener(this);
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
	
	public static void main(String[] args){
//		Vector<Boid> bandada = new Vector<Boid>();
//		Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
		PanelMuestraBoids ventana = new PanelMuestraBoids();
		ventana.setTamanoBan(30);
		//Bucle para crear la bandada
		for (int j = 0;j<ventana.getTamanoBan();j++){
			double posAux[] = {Math.random()*800,Math.random()};
			double velAux[] = {Math.random(),Math.random()};
			Matrix posi = new Matrix(posAux,2);
			Matrix vel = new Matrix(velAux,2);			
			ventana.getBandada().add(new Boid(posi,vel));				
//			ventana.getObstaculos().add(new Obstaculo(posiObs,velObs));
		}
		
		
		ventana.pintor.introducirBandada(ventana.getBandada());
//		ventana.pintor.introducirObstaculos(ventana.getObstaculos());
		Console.run(ventana,1200,1000);
		int alturaPanel = ventana.pintor.getHeight();
		int anchuraPanel = ventana.pintor.getWidth();
		System.out.println(alturaPanel+ "  "+anchuraPanel);
//		Bucle para crear los bordes
		int cont = 1;
		for (int k=0; k<2;k++){			
			for (int i = 0;i < alturaPanel;i=i+10){
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
			for (int i = 0;i < anchuraPanel;i=i+10){
				double posObstaculos[] = {i,cont};
				double velObstaculos[] = {0,0};				
				Matrix posiObs = new Matrix(posObstaculos,2);
				Matrix velObs = new Matrix(velObstaculos,2);
				ventana.getObstaculos().add(new Obstaculo(posiObs,velObs));
			}
			cont = alturaPanel-15;
		}
		System.out.println("cont = "+cont);
		ventana.pintor.introducirObstaculos(ventana.getObstaculos());
		double distMin = Double.POSITIVE_INFINITY;
		int indMin = 0;
		int indMinAnt = 0;
		while(true){
			while(ventana.play){
				for (int j = 0;j<ventana.getTamanoBan();j++){
					ventana.getBandada().elementAt(j).mover(ventana.getBandada()
							,ventana.getObstaculos(),j,Boid.getObjetivo());
					double dist = ventana.getBandada().elementAt(j).getDistObjetivo();
					if (dist < distMin){
						distMin = dist;
						indMin = j;
					}
				}
				//El lider será el que se encuentre más cerca del objetivo
				ventana.getBandada().elementAt(indMinAnt).setLider(false);
				ventana.getBandada().elementAt(indMin).setLider(true);
				indMinAnt = indMin;
				distMin = Double.POSITIVE_INFINITY;
				ventana.pintor.repaint();
//				try {
//	            	Thread.sleep(50);
//	        	} catch (Exception e) {
//	        	}
			}
		}
		
	}

	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spinnerCohesion){
			Boid.setCohesion(spCohesion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerSeparacion){
			Boid.setSeparacion(spSeparacion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerAlineacion){
			Boid.setAlineacion(spAlineacion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerObjetivo){
			Boid.setVelObjetivo(spObjetivo.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerEvitaObs){
			Boid.setEvitaObstaculo(spEvitaObs.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerVelMax){
			Boid.setVelMax(spVelMax.getNumber().doubleValue());
		}
	}

	public void actionPerformed(ActionEvent e){
		if (e.getSource() == pausa){
			if (!play){ // La etiqueta del botón cambia
				pausa.setText("Pausa");
			}
			if (play){
				pausa.setText("Play");
			}
			play = !play;
		}
		if (e.getSource() == colocarObs){
			if (!colocandoObs){ // La etiqueta del botón cambia
				colocarObs.setText("Colocando obstáculos");				
				colocarBan.setEnabled(false);
			}
			if (colocandoObs){
				colocarObs.setText("Pulse para colocar obstáculos");
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
				colocarBan.setText("Pulse para colocar la Bandada");
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
				double pos[] = {e.getX()+Math.random()*getTamanoBan()*2, e.getY()+Math.random()*getTamanoBan()*2};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {0,0};
				Matrix velo = new Matrix(vel,2);
				this.getBandada().elementAt(i).getForma().transform(AffineTransform.getTranslateInstance(pos[0]-getBandada().elementAt(i).getPosicion().get(0,0),
						pos[1]-getBandada().elementAt(i).getPosicion().get(1,0)));
				this.getBandada().elementAt(i).setPosicion(posi);			
				this.getBandada().elementAt(i).setVelocidad(velo);
			}
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
		super.paintComponent(g2);
		// Pinto los Boids
		for (int i=0;i<bandadaPintar.size();i++){
			if (bandadaPintar.elementAt(i).isLider())
				g2.setColor(Color.green);
			else
				g2.setColor(Color.blue);
			g2.draw(bandadaPintar.elementAt(i).getForma());
			g2.fill(bandadaPintar.elementAt(i).getForma());
		}
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
