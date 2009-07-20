/**
 * 
 */
package sibtra.ui;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.Modulo;
import sibtra.ui.defs.Motor;
import sibtra.util.CargadorDeModulos;
import sibtra.util.ClasesEnPaquete;

/**
 * Panel que permitira la selección de los módulos básicos
 * @author alberto
 *
 */
@SuppressWarnings("serial")
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
	AccionCrear accionCrear;
	AccionActivar accionActivar;
	AccionParar accionParar;
	AccionBorrar accionBorrar;
	AccionRefrescar accionRefrescar;
	public Motor obMotor;
	public CalculoDireccion obDireccion;
	public CalculoVelocidad obVelocidad;
	public DetectaObstaculos[] obsDetec;
	ClassLoader cargadorClases;
	private DefaultListModel modeloLista;

	public PanelEligeModulos(VentanasMonitoriza ventMonito) {
		super(new GridLayout(0,2,10,10));
		
		ventanaMonitoriza=ventMonito;
		
		
		//Para seleccionar los 3 tipos de módulos
		add(new JLabel("Motor",JLabel.TRAILING));
		jcombMotor=new JComboBox();
		add(jcombMotor);
		
		add(new JLabel("Calculo Direccion",JLabel.TRAILING));
		jcombDireccion=new JComboBox();
		add(jcombDireccion);
		
		add(new JLabel("Calculo Velocidad",JLabel.TRAILING));
		jcombVelocidad=new JComboBox();
		add(jcombVelocidad);
		
		add(new JLabel("Detector obstaculos",JLabel.TRAILING));
		modeloLista=new DefaultListModel();
		jcombDetecObstaculos=new JList(modeloLista);
		add(jcombDetecObstaculos);
		
		refrescaModulos();
		
		//Instanciamos las Acciones
		accionCrear=new AccionCrear();
		accionActivar=new AccionActivar();
		accionParar=new AccionParar();
		accionBorrar=new AccionBorrar();
		accionRefrescar=new AccionRefrescar();
		
		//ponemos los botones de las acciones
        ventanaMonitoriza.menuAcciones.addSeparator();
		add(new JButton(accionCrear));
		add(new JButton(accionActivar));
		add(new JButton(accionParar));
		add(new JButton(accionBorrar));
		add(new JButton(accionRefrescar));
		
		//ponemos los botones tambien en el menu de opciones
		ventanaMonitoriza.menuAcciones.add(accionActivar);
		ventanaMonitoriza.menuAcciones.add(accionParar);
		ventanaMonitoriza.menuAcciones.add(accionCrear);
		ventanaMonitoriza.menuAcciones.add(accionBorrar);
		ventanaMonitoriza.menuAcciones.add(accionRefrescar);
		
		
	}
	
	protected void refrescaModulos() {
		
		cargadorClases = new CargadorDeModulos(CargadorDeModulos.class.getClassLoader(),"sibtra.ui.modulos");

		arrClasMotor=ClasesEnPaquete.clasesImplementan("sibtra.ui.defs.Motor", "sibtra.ui.modulos",cargadorClases);
		String[] arrNomClasMotor=ClasesEnPaquete.nombreClases(arrClasMotor);
		jcombMotor.removeAllItems();
		for(String sa: arrNomClasMotor)
			jcombMotor.addItem(sa);
		
		arrClasDir=ClasesEnPaquete.clasesImplementan("sibtra.ui.defs.CalculoDireccion", "sibtra.ui.modulos",cargadorClases);
		String[] arrNomClasDir=ClasesEnPaquete.nombreClases(arrClasDir);
		jcombDireccion.removeAllItems();
		for(String sa: arrNomClasDir)
			jcombDireccion.addItem(sa);
			
		arrClasVel=ClasesEnPaquete.clasesImplementan("sibtra.ui.defs.CalculoVelocidad", "sibtra.ui.modulos",cargadorClases);
		String[] arrNomClasVel=ClasesEnPaquete.nombreClases(arrClasVel);
		jcombVelocidad.removeAllItems();
		for(String sa: arrNomClasVel)
			jcombVelocidad.addItem(sa);
			
		arrClasDectObs=ClasesEnPaquete.clasesImplementan("sibtra.ui.defs.DetectaObstaculos", "sibtra.ui.modulos",cargadorClases);
		String[] arrNomClasDectObs=ClasesEnPaquete.nombreClases(arrClasDectObs);
		modeloLista.removeAllElements();
		for(String sa: arrNomClasDectObs)
			modeloLista.addElement(sa);
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
	
	
	class AccionRefrescar extends AbstractAction {
		
		public AccionRefrescar() {
			super("Refresca Modulos");
			setEnabled(true);
		}

		public void actionPerformed(ActionEvent e) {
			refrescaModulos();
			PanelEligeModulos.this.repaint();
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
