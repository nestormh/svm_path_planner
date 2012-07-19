package repgrafica;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import boids.Boid;
import boids.DesignPoint;

import com.bruceeckel.swing.Console;

public class ConfigParam extends JApplet implements ActionListener, ChangeListener{
	
//	int numBoidsOk = 2;
//	SpinnerNumberModel spNumBoidsOk = new SpinnerNumberModel(numBoidsOk,0,1000,1);
//	JSpinner spinnerNumBoidsOk = new JSpinner(spNumBoidsOk);	
//	JLabel numBoidsOkLabel = new JLabel("Número de boids con éxito");
	double tMax = 10;
	SpinnerNumberModel spTmax = new SpinnerNumberModel(tMax,0,1000,1);
	JSpinner spinnerTmax = new JSpinner(spTmax);
	JLabel tMaxLabel = new JLabel("Tiempo maximo para la simulacion");
	JButton configurar = new JButton("Configurar");
	
	Vector <Parametro> params;
	Vector <Hashtable> vectorSim = new Vector<Hashtable>();
	String[] nomParam = {"Radio Obstaculos","Radio Cohesion","Radio Separacion"
			,"Radio Alineacion","Peso Cohesion","Peso Separacion","Peso Alineacion"
			,"Peso Objetivo","Peso Obstaculo","Peso Lider","Velocidad Maxima","N� Boids","Boids Ok"};
	double[] valorParam = {Boid.getRadioObstaculo(),Boid.getRadioCohesion(),Boid.getRadioSeparacion()
			,Boid.getRadioAlineacion(),Boid.getPesoCohesion(),Boid.getPesoSeparacion(),Boid.getPesoAlineacion()
			,Boid.getPesoObjetivo(),Boid.getPesoObstaculo(),Boid.getPesoLider(),Boid.getVelMax(),20
			,1};
	
	// Inicialización gráfica
	public void init(){
		params = new Vector <Parametro>();
		Container cp = getContentPane();
		
		// Definición del panel de datos
		JPanel panelDatos = new JPanel(new GridLayout(nomParam.length,1));
		for (int i=0;i<nomParam.length;i++){
			params.add(new Parametro(nomParam[i],valorParam[i],valorParam[i],1));			
			panelDatos.add(params.elementAt(i));
		}
		cp.add(panelDatos);
		
		// Definición del panel de parámetros de la simulación
		JPanel panelSur = new JPanel(new FlowLayout());
//		panelSur.add(spinnerNumBoidsOk);
//		spinnerNumBoidsOk.addChangeListener(this);
//		panelSur.add(numBoidsOkLabel);
		panelSur.add(spinnerTmax);
		spinnerTmax.addChangeListener(this);
		panelSur.add(tMaxLabel);
		panelSur.add(configurar);
		configurar.addActionListener(this);
		cp.add(BorderLayout.SOUTH,panelSur);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == configurar){
			vectorSim.clear();
			for (int i=0;i<params.size();i++){
				if (params.elementAt(i).isSelected()){
					Vector <Hashtable> vecAux =new Vector <Hashtable>();
					if (vectorSim.size() <= 0){
						double valorParam = params.elementAt(i).getValorIni();
						while (valorParam <= params.elementAt(i).getValorFin()){
							Hashtable nuevoPunto = new Hashtable();
							nuevoPunto.put(params.elementAt(i).getNombre(),valorParam);
							vecAux.add(nuevoPunto);	
							valorParam = valorParam + params.elementAt(i).getValorPaso();
						}						
					}// Cuidado con los punteros!!!!!!!!!!!
					else{
						for (int j=0;j<vectorSim.size();j++){
							double valorParam = params.elementAt(i).getValorIni();
							while (valorParam <= params.elementAt(i).getValorFin()){
								Hashtable nuevoPunto = new Hashtable();
								for (Enumeration enu = vectorSim.elementAt(j).keys() ; enu.hasMoreElements() ;) {									
									String nombre = (String)enu.nextElement();
							        nuevoPunto.put(nombre,vectorSim.elementAt(j).get(nombre));							        
								}
								nuevoPunto.put(params.elementAt(i).getNombre(),valorParam);
								vecAux.add(nuevoPunto);
								valorParam = valorParam + params.elementAt(i).getValorPaso();
							}						
						}
					}
					vectorSim = vecAux;
//					System.out.println("Tamaño vectorSim " + vectorSim.size());
//					for(int k=0;k<vectorSim.size();k++){
//						for (Enumeration enu = vectorSim.elementAt(k).keys() ; enu.hasMoreElements() ;) {
//					         String nombre = (String)enu.nextElement();
//					         System.out.println("-------");
//							 System.out.println("Parámetro " + nombre);
//							 System.out.println("Valor " + vectorSim.elementAt(k).get(nombre));
//							 System.out.println("-------");
//						}												
//					}
				
				}				
			}
			
		}		
	}
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spinnerTmax){
			setTMax(spTmax.getNumber().doubleValue());
		}
//		if (e.getSource() == spinnerNumBoidsOk){
//			setNumBoidsOk(spNumBoidsOk.getNumber().intValue());
//		}
		
	}
	
	public static void main(String[] args){
		ConfigParam configurador = new ConfigParam();
		Console.run(configurador,1000,300);
	}

	public String[] getNomParam() {
		return nomParam;
	}

	public Vector<Hashtable> getVectorSim() {
		return vectorSim;
	}

//	public int getNumBoidsOk() {
//		return numBoidsOk;
//	}
//
//	public void setNumBoidsOk(int numBoidsOk) {
//		this.numBoidsOk = numBoidsOk;
//	}

	public double getTMax() {
		return tMax;
	}

	public void setTMax(double max) {
		tMax = max;
	}
}
