/**
 * 
 */
package sibtra.ui;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import sibtra.ui.modulos.CalculoDireccion;
import sibtra.ui.modulos.CalculoVelocidad;
import sibtra.ui.modulos.DetectaObstaculos;
import sibtra.ui.modulos.Modulo;
import sibtra.ui.modulos.Motor;
import sibtra.util.ClasesEnPaquete;

/**
 * Panel que permitira la selección de los módulos básicos
 * @author alberto
 *
 */
public class PanelEligeModulos extends JPanel {
	
	private Class[] arrClasMotor;
	private JComboBox jcombMotor;
	private Class[] arrClasDir;
	private Class[] arrClasVel;
	private Class[] arrClasDectObs;
	private JComboBox jcombDireccion;
	private JComboBox jcombVelocidad;
	private JList jcombDetecObstaculos;
	private VentanasMonitoriza ventanaMonitoriza;
	private AccionCrear accionCrear;
	private AccionActivar accionActivar;
	private AccionParar accionParar;
	private AccionBorrar accionBorrar;
	public Motor obMotor;
	public CalculoDireccion obDireccion;
	public CalculoVelocidad obVelocidad;
	public DetectaObstaculos[] obsDetec;

	public PanelEligeModulos(VentanasMonitoriza ventMonito) {
		super(new GridLayout(0,2,10,10));
		
		ventanaMonitoriza=ventMonito;
		
		arrClasMotor=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.Motor", "sibtra.ui.modulos");
		String[] arrNomClasMotor=ClasesEnPaquete.nombreClases(arrClasMotor);
			
		arrClasDir=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.CalculoDireccion", "sibtra.ui.modulos");
		String[] arrNomClasDir=ClasesEnPaquete.nombreClases(arrClasDir);
			
		arrClasVel=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.CalculoVelocidad", "sibtra.ui.modulos");
		String[] arrNomClasVel=ClasesEnPaquete.nombreClases(arrClasVel);
			
		arrClasDectObs=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.DetectaObstaculos", "sibtra.ui.modulos");
		String[] arrNomClasDectObs=ClasesEnPaquete.nombreClases(arrClasDectObs);
			
		
		//Para seleccionar los 3 tipos de módulos
		add(new JLabel("Motor",JLabel.TRAILING));
		jcombMotor=new JComboBox(arrNomClasMotor);
		add(jcombMotor);
		
		add(new JLabel("Calculo Direccion",JLabel.TRAILING));
		jcombDireccion=new JComboBox(arrNomClasDir);
		add(jcombDireccion);
		
		add(new JLabel("Calculo Velocidad",JLabel.TRAILING));
		jcombVelocidad=new JComboBox(arrNomClasVel);
		add(jcombVelocidad);
		
		add(new JLabel("Detector obstaculos",JLabel.TRAILING));
		jcombDetecObstaculos=new JList(arrNomClasDectObs);
		add(jcombDetecObstaculos);
		
		//Instanciamos las Acciones
		accionCrear=new AccionCrear();
		accionActivar=new AccionActivar();
		accionParar=new AccionParar();
		accionBorrar=new AccionBorrar();
		
		//ponemos los botones de las acciones
		add(new JButton(accionCrear));
		add(new JButton(accionActivar));
		add(new JButton(accionParar));
		add(new JButton(accionBorrar));
		
	}
	
	class AccionCrear extends AbstractAction {
		
		public AccionCrear() {
			super("Crear Modulos");
			setEnabled(true);
		}

		/** Instancia los objetos de los 4 tipos si hay alguno repetido solo instancia un objeto */
		public void actionPerformed(ActionEvent e) {
			//primero creamos todos los objetos
			try {
				//Tipo Motor
				int indMot=jcombMotor.getSelectedIndex();
				if(indMot<0 || indMot>=arrClasMotor.length) {
					System.err.println(getClass().getName()+": indice para motor fuera de rango");
					return;
				}
				Class<Motor> clasMot=arrClasMotor[indMot];
				obMotor=(Motor)clasMot.newInstance();
				
				//Tipo Direccion
				int indDir=jcombDireccion.getSelectedIndex();
				if(indDir<0 || indDir>=arrClasDir.length) {
					System.err.println(getClass().getName()+": indice para direccion fuera de rango");
					return;
				}
				if(arrClasDir[indDir].equals(arrClasMotor[indMot])) {
					System.out.println(getClass().getName()+": clase de motor y dirección son la misma");
					obDireccion=(CalculoDireccion)obMotor;
				} else {
					//clases distintas, tenemos que crear objeto
					Class<CalculoDireccion> clasDir=arrClasDir[indDir];				
					obDireccion=(CalculoDireccion)clasDir.newInstance();
				}
				
				//Tipo Velocidad
				int indVel=jcombVelocidad.getSelectedIndex();
				if(indVel<0 || indVel>=arrClasDir.length) {
					System.err.println(getClass().getName()+": indice para Velocidad fuera de rango");
					return;
				}
				if(arrClasVel[indVel].equals(arrClasMotor[indMot])) {
					System.out.println(getClass().getName()+": clase de motor y velocidad son la misma");
					obVelocidad=(CalculoVelocidad)obMotor;
				} else if(arrClasVel[indVel].equals(arrClasDir[indDir])) {
					System.out.println(getClass().getName()+": clase de direccion y velocidad son la misma");
					obVelocidad=(CalculoVelocidad)obDireccion;
				} else {
					//clases distintas, tenemos que crear objeto
					Class<CalculoVelocidad> clasVel=arrClasVel[indVel];				
					obVelocidad=(CalculoVelocidad)clasVel.newInstance();
				}
				
				int[] decSel=jcombDetecObstaculos.getSelectedIndices();
				obsDetec=new DetectaObstaculos[decSel.length]; //array para acoger objetos detectores
				System.out.println(getClass().getName()+": Hay "+decSel.length+" detectores seleccionados");
				for(int ids=0;ids<decSel.length;ids++){
					int i=decSel[ids];
					if(arrClasDectObs[i].equals(arrClasMotor[indMot])) {
						System.out.println(getClass().getName()+": clase de motor y detector "+i+" son la misma");
						obsDetec[ids]=(DetectaObstaculos)obMotor;
					} else if(arrClasDectObs[i].equals(arrClasDir[indDir])) {
						System.out.println(getClass().getName()+": clase de direccion y y detector "+i+" son la misma");
						obsDetec[ids]=(DetectaObstaculos)obDireccion;
					} else if(arrClasDectObs[i].equals(arrClasVel[indVel])) {
						System.out.println(getClass().getName()+": clase de velocidad y detector "+i+"  son la misma");
						obsDetec[ids]=(DetectaObstaculos)obVelocidad;
					} else {
						//clases distintas, tenemos que crear objeto
						Class<DetectaObstaculos> clasDet=arrClasDectObs[i];				
						obsDetec[ids]=(DetectaObstaculos)clasDet.newInstance();
					}
				}
								
			} catch (Exception excep) {
				System.err.println(getClass().getName()+": problemas en el proceso de creacion "+excep.getMessage());
				excep.printStackTrace();
			}
			// Tenemos los objetos, procedemos a inicializarlos Comprobando si hay algún problema
			// si alguna inicialización falla, terminamos los módulos inicializados y salimos
			if(!obMotor.setVentanaMonitoriza(ventanaMonitoriza)) {
				avisaFalloMoudulo(obMotor);
				obsDetec=null;
				obVelocidad=null;
				obDireccion=null;
				obMotor=null;
				return;
			}
			if(!obDireccion.setVentanaMonitoriza(ventanaMonitoriza)) {
				avisaFalloMoudulo(obDireccion);
				obMotor.terminar();
				obsDetec=null;
				obVelocidad=null;
				obDireccion=null;
				obMotor=null;
				return;
			}
			if(!obVelocidad.setVentanaMonitoriza(ventanaMonitoriza)) {
				avisaFalloMoudulo(obVelocidad);
				obDireccion.terminar();
				obMotor.terminar();
				obsDetec=null;
				obVelocidad=null;
				obDireccion=null;
				obMotor=null;
				return;
			}
			for(int i=0; i<obsDetec.length;i++) {
				if(!obsDetec[i].setVentanaMonitoriza(ventanaMonitoriza))
				{
					avisaFalloMoudulo(obsDetec[i]);
					for(int j=0;j<i;j++)
						obsDetec[j].terminar();
					obVelocidad.terminar();
					obDireccion.terminar();
					obMotor.terminar();
					obsDetec=null;
					obVelocidad=null;
					obDireccion=null;
					obMotor=null;
					return;
				}
			}
			//le comunicamos los modulos al motor
			obMotor.setCalculadorDireccion(obDireccion);
			obMotor.setCalculadorVelocidad(obVelocidad);
			obMotor.setDetectaObstaculos(obsDetec);

			//terminamos cambiando las habilitaciones
			this.setEnabled(false);
			accionActivar.setEnabled(true);
			accionBorrar.setEnabled(true);
		}
		
		/** Saca ventana avisando fallo del arranque */
		private void avisaFalloMoudulo(Modulo mod) {
		JOptionPane.showMessageDialog(ventanaMonitoriza.ventanaPrincipal,
			    "No se inicializamo correctamente el módulo "+mod.getNombre()
			    +"\nSe cancela la creacion",
			    "Fallo arranque",
			    JOptionPane.ERROR_MESSAGE);
		}
	}
	
	class AccionActivar extends AbstractAction {
		
		public AccionActivar() {
			super("Activar Modulos");
			setEnabled(false);
		}

		public void actionPerformed(ActionEvent e) {
			obMotor.actuar();
			setEnabled(false);
			accionParar.setEnabled(true);
			accionBorrar.setEnabled(false); //solo puede borrar si parado
		}
	}
	
	class AccionParar extends AbstractAction {
		
		public AccionParar() {
			super("Parar Modulos");
			setEnabled(false);
		}

		public void actionPerformed(ActionEvent e) {
			obMotor.parar();
			setEnabled(false);
			accionActivar.setEnabled(true);
			accionBorrar.setEnabled(true); //parado, puede borrar
		}
	}
	
	
	class AccionBorrar extends AbstractAction {
		
		public AccionBorrar() {
			super("Borra Modulos");
			setEnabled(false);
		}

		public void actionPerformed(ActionEvent e) {
			//Le decimos a todos los modulos que terminen
			obMotor.parar();
			for(DetectaObstaculos doa: obsDetec)
				doa.terminar();
			obVelocidad.terminar();
			obDireccion.terminar();
			obMotor.terminar();
			obsDetec=null;
			obVelocidad=null;
			obDireccion=null;
			obMotor=null;
			//cambio en habilitaciones
			setEnabled(false);
			accionCrear.setEnabled(true);
			accionActivar.setEnabled(false);
			accionParar.setEnabled(false);
		}
	}
}
